-- PostgreSQL model for hospital bed state transitions.
--
-- The design separates physical bed identity from occupancy episodes. Every
-- state transition is recorded as an immutable event and a trigger maintains a
-- current-state projection while enforcing allowed transitions.

BEGIN;

CREATE SCHEMA IF NOT EXISTS bedmgmt;

CREATE TYPE bedmgmt.bed_state AS ENUM (
    'available',
    'reserved',
    'occupied',
    'cleaning',
    'blocked',
    'maintenance'
);

CREATE TABLE bedmgmt.units (
    unit_id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    unit_code        text NOT NULL UNIQUE,
    unit_name        text NOT NULL,
    specialty        text NOT NULL
);

CREATE TABLE bedmgmt.beds (
    bed_id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    unit_id          bigint NOT NULL REFERENCES bedmgmt.units(unit_id),
    bed_code         text NOT NULL,
    room_code        text NOT NULL,
    current_state    bedmgmt.bed_state NOT NULL DEFAULT 'available',
    current_patient  bigint,
    version          bigint NOT NULL DEFAULT 0,
    UNIQUE(unit_id, bed_code)
);

CREATE TABLE bedmgmt.bed_events (
    event_id         bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    bed_id           bigint NOT NULL REFERENCES bedmgmt.beds(bed_id),
    sequence_no      bigint NOT NULL,
    from_state       bedmgmt.bed_state NOT NULL,
    to_state         bedmgmt.bed_state NOT NULL,
    patient_id       bigint,
    reason_code      text NOT NULL,
    actor_id         text NOT NULL,
    occurred_at      timestamptz NOT NULL DEFAULT clock_timestamp(),
    details          jsonb NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE(bed_id, sequence_no)
);

CREATE INDEX bed_events_bed_time_idx
    ON bedmgmt.bed_events(bed_id, occurred_at DESC);

CREATE TABLE bedmgmt.allowed_transitions (
    from_state bedmgmt.bed_state NOT NULL,
    to_state   bedmgmt.bed_state NOT NULL,
    PRIMARY KEY(from_state, to_state)
);

INSERT INTO bedmgmt.allowed_transitions(from_state, to_state) VALUES
    ('available', 'reserved'),
    ('available', 'occupied'),
    ('available', 'blocked'),
    ('available', 'maintenance'),
    ('reserved', 'available'),
    ('reserved', 'occupied'),
    ('reserved', 'blocked'),
    ('occupied', 'cleaning'),
    ('occupied', 'blocked'),
    ('cleaning', 'available'),
    ('cleaning', 'blocked'),
    ('blocked', 'available'),
    ('blocked', 'cleaning'),
    ('blocked', 'maintenance'),
    ('maintenance', 'available'),
    ('maintenance', 'blocked')
ON CONFLICT DO NOTHING;

CREATE OR REPLACE FUNCTION bedmgmt.transition_bed(
    p_bed_id bigint,
    p_expected_version bigint,
    p_to_state bedmgmt.bed_state,
    p_reason_code text,
    p_actor_id text,
    p_patient_id bigint DEFAULT NULL,
    p_details jsonb DEFAULT '{}'::jsonb
)
RETURNS bedmgmt.bed_events
LANGUAGE plpgsql
AS $$
DECLARE
    v_bed bedmgmt.beds%ROWTYPE;
    v_event bedmgmt.bed_events%ROWTYPE;
    v_next_patient bigint;
BEGIN
    SELECT *
      INTO v_bed
      FROM bedmgmt.beds
     WHERE bed_id = p_bed_id
     FOR UPDATE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'bed % not found', p_bed_id;
    END IF;

    IF v_bed.version <> p_expected_version THEN
        RAISE EXCEPTION
            'concurrency conflict for bed %, expected version %, actual %',
            p_bed_id, p_expected_version, v_bed.version
            USING ERRCODE = '40001';
    END IF;

    IF NOT EXISTS (
        SELECT 1
          FROM bedmgmt.allowed_transitions
         WHERE from_state = v_bed.current_state
           AND to_state = p_to_state
    ) THEN
        RAISE EXCEPTION 'transition % -> % is not allowed',
            v_bed.current_state, p_to_state;
    END IF;

    IF p_reason_code IS NULL OR btrim(p_reason_code) = '' THEN
        RAISE EXCEPTION 'reason_code is required';
    END IF;

    IF p_actor_id IS NULL OR btrim(p_actor_id) = '' THEN
        RAISE EXCEPTION 'actor_id is required';
    END IF;

    -- Patient identity must be explicit for occupied state and cleared only
    -- through a transition that ends the occupancy episode.
    IF p_to_state = 'occupied' AND p_patient_id IS NULL THEN
        RAISE EXCEPTION 'patient_id is required when occupying a bed';
    END IF;

    IF p_to_state IN ('available', 'cleaning', 'blocked', 'maintenance') THEN
        v_next_patient := NULL;
    ELSIF p_to_state = 'reserved' THEN
        v_next_patient := p_patient_id;
    ELSE
        v_next_patient := p_patient_id;
    END IF;

    INSERT INTO bedmgmt.bed_events(
        bed_id,
        sequence_no,
        from_state,
        to_state,
        patient_id,
        reason_code,
        actor_id,
        details
    )
    VALUES (
        v_bed.bed_id,
        v_bed.version + 1,
        v_bed.current_state,
        p_to_state,
        p_patient_id,
        p_reason_code,
        p_actor_id,
        COALESCE(p_details, '{}'::jsonb)
    )
    RETURNING * INTO v_event;

    UPDATE bedmgmt.beds
       SET current_state = p_to_state,
           current_patient = v_next_patient,
           version = v_bed.version + 1
     WHERE bed_id = v_bed.bed_id;

    RETURN v_event;
END;
$$;

-- Convenience view for operational dashboards. The view does not infer
-- availability from patient assignments; it reports the explicit state machine.
CREATE OR REPLACE VIEW bedmgmt.current_capacity AS
SELECT
    u.unit_id,
    u.unit_code,
    u.unit_name,
    COUNT(*) AS total_beds,
    COUNT(*) FILTER (WHERE b.current_state = 'available') AS available_beds,
    COUNT(*) FILTER (WHERE b.current_state = 'reserved') AS reserved_beds,
    COUNT(*) FILTER (WHERE b.current_state = 'occupied') AS occupied_beds,
    COUNT(*) FILTER (WHERE b.current_state = 'cleaning') AS cleaning_beds,
    COUNT(*) FILTER (WHERE b.current_state = 'blocked') AS blocked_beds,
    COUNT(*) FILTER (WHERE b.current_state = 'maintenance') AS maintenance_beds
FROM bedmgmt.units AS u
JOIN bedmgmt.beds AS b ON b.unit_id = u.unit_id
GROUP BY u.unit_id, u.unit_code, u.unit_name;

-- Occupancy episodes are reconstructed from events so historical reports do
-- not depend on the mutable current_state column.
CREATE OR REPLACE VIEW bedmgmt.occupancy_episodes AS
WITH occupied_events AS (
    SELECT
        e.*,
        LEAD(e.occurred_at) OVER (
            PARTITION BY e.bed_id
            ORDER BY e.sequence_no
        ) AS next_event_at,
        LEAD(e.to_state) OVER (
            PARTITION BY e.bed_id
            ORDER BY e.sequence_no
        ) AS next_state
    FROM bedmgmt.bed_events AS e
),
starts AS (
    SELECT
        event_id,
        bed_id,
        patient_id,
        occurred_at AS occupied_at,
        next_event_at,
        next_state
    FROM occupied_events
    WHERE to_state = 'occupied'
)
SELECT
    s.event_id AS occupancy_start_event_id,
    s.bed_id,
    s.patient_id,
    s.occupied_at,
    (
        SELECT MIN(e2.occurred_at)
        FROM bedmgmt.bed_events AS e2
        WHERE e2.bed_id = s.bed_id
          AND e2.sequence_no > (
              SELECT e1.sequence_no
              FROM bedmgmt.bed_events AS e1
              WHERE e1.event_id = s.event_id
          )
          AND e2.to_state IN ('cleaning', 'blocked', 'available', 'maintenance')
    ) AS released_at
FROM starts AS s;

-- Identify beds whose current projection disagrees with the last event.
CREATE OR REPLACE VIEW bedmgmt.projection_integrity AS
WITH last_event AS (
    SELECT DISTINCT ON (bed_id)
        bed_id,
        sequence_no,
        to_state,
        patient_id
    FROM bedmgmt.bed_events
    ORDER BY bed_id, sequence_no DESC
)
SELECT
    b.bed_id,
    b.current_state,
    b.current_patient,
    b.version,
    le.sequence_no AS last_sequence,
    le.to_state AS last_event_state,
    le.patient_id AS last_event_patient,
    (
        b.version = COALESCE(le.sequence_no, 0)
        AND (
            le.to_state IS NULL
            OR b.current_state = le.to_state
        )
    ) AS projection_consistent
FROM bedmgmt.beds AS b
LEFT JOIN last_event AS le ON le.bed_id = b.bed_id;

-- Example seed and workflow.
INSERT INTO bedmgmt.units(unit_code, unit_name, specialty)
VALUES ('4W', 'Fourth West', 'General Medicine')
ON CONFLICT (unit_code) DO NOTHING;

INSERT INTO bedmgmt.beds(unit_id, bed_code, room_code)
SELECT unit_id, v.bed_code, v.room_code
FROM bedmgmt.units
CROSS JOIN (
    VALUES ('401-A', '401'), ('401-B', '401'), ('402-A', '402')
) AS v(bed_code, room_code)
WHERE unit_code = '4W'
ON CONFLICT (unit_id, bed_code) DO NOTHING;

DO $$
DECLARE
    v_id bigint;
    v_version bigint;
BEGIN
    SELECT bed_id, version
      INTO v_id, v_version
      FROM bedmgmt.beds
     WHERE bed_code = '401-A';

    IF v_version = 0 THEN
        PERFORM bedmgmt.transition_bed(
            v_id, 0, 'reserved', 'ED_ADMISSION',
            'scheduler:demo', 880021,
            '{"source":"emergency_department"}'::jsonb
        );
        PERFORM bedmgmt.transition_bed(
            v_id, 1, 'occupied', 'PATIENT_ARRIVED',
            'nurse:demo', 880021,
            '{"transport":"wheelchair"}'::jsonb
        );
        PERFORM bedmgmt.transition_bed(
            v_id, 2, 'cleaning', 'DISCHARGED',
            'nurse:demo', NULL,
            '{"terminal_clean":false}'::jsonb
        );
        PERFORM bedmgmt.transition_bed(
            v_id, 3, 'available', 'CLEAN_COMPLETE',
            'environmental:demo', NULL,
            '{"inspection":"passed"}'::jsonb
        );
    END IF;
END;
$$;

COMMIT;
