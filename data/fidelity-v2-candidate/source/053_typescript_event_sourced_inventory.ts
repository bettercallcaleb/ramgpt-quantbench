/**
 * Event-sourced inventory service component.
 *
 * The aggregate stores no mutable database row as its source of truth. Current
 * state is reconstructed by folding immutable events. Commands validate against
 * the reconstructed state, then append one or more new events.
 */

type Sku = string;
type WarehouseId = string;
type EventId = string;

interface EventEnvelope<T extends InventoryEvent = InventoryEvent> {
  id: EventId;
  streamId: string;
  sequence: number;
  occurredAt: string;
  payload: T;
}

type InventoryEvent =
  | { type: "StockReceived"; sku: Sku; warehouse: WarehouseId; quantity: number; reference: string }
  | { type: "StockReserved"; sku: Sku; warehouse: WarehouseId; quantity: number; orderId: string }
  | { type: "ReservationReleased"; sku: Sku; warehouse: WarehouseId; quantity: number; orderId: string }
  | { type: "StockShipped"; sku: Sku; warehouse: WarehouseId; quantity: number; orderId: string }
  | { type: "StockAdjusted"; sku: Sku; warehouse: WarehouseId; delta: number; reason: string };

interface InventoryState {
  sku: Sku;
  warehouse: WarehouseId;
  onHand: number;
  reserved: number;
  shipped: number;
  reservations: Map<string, number>;
  version: number;
}

function emptyState(sku: Sku, warehouse: WarehouseId): InventoryState {
  return {
    sku,
    warehouse,
    onHand: 0,
    reserved: 0,
    shipped: 0,
    reservations: new Map(),
    version: 0,
  };
}

function assertPositiveInteger(value: number, label: string): void {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${label} must be a positive integer`);
  }
}

function apply(state: InventoryState, event: InventoryEvent): InventoryState {
  const next: InventoryState = {
    ...state,
    reservations: new Map(state.reservations),
    version: state.version + 1,
  };

  switch (event.type) {
    case "StockReceived":
      next.onHand += event.quantity;
      break;

    case "StockReserved": {
      next.reserved += event.quantity;
      next.reservations.set(
        event.orderId,
        (next.reservations.get(event.orderId) ?? 0) + event.quantity,
      );
      break;
    }

    case "ReservationReleased": {
      next.reserved -= event.quantity;
      const remaining = (next.reservations.get(event.orderId) ?? 0) - event.quantity;
      if (remaining > 0) next.reservations.set(event.orderId, remaining);
      else next.reservations.delete(event.orderId);
      break;
    }

    case "StockShipped": {
      next.onHand -= event.quantity;
      next.reserved -= event.quantity;
      next.shipped += event.quantity;
      const remaining = (next.reservations.get(event.orderId) ?? 0) - event.quantity;
      if (remaining > 0) next.reservations.set(event.orderId, remaining);
      else next.reservations.delete(event.orderId);
      break;
    }

    case "StockAdjusted":
      next.onHand += event.delta;
      break;

    default: {
      const neverEvent: never = event;
      throw new Error(`unknown event ${(neverEvent as any).type}`);
    }
  }

  if (next.onHand < 0 || next.reserved < 0 || next.reserved > next.onHand) {
    throw new Error(`invalid inventory state after ${event.type}`);
  }

  return next;
}

function rebuild(
  sku: Sku,
  warehouse: WarehouseId,
  events: readonly EventEnvelope[],
): InventoryState {
  return events.reduce(
    (state, envelope) => apply(state, envelope.payload),
    emptyState(sku, warehouse),
  );
}

class ConcurrencyError extends Error {}
class DomainError extends Error {}

interface AppendOptions {
  expectedVersion: number;
}

interface EventStore {
  read(streamId: string): Promise<EventEnvelope[]>;
  append(streamId: string, events: InventoryEvent[], options: AppendOptions): Promise<EventEnvelope[]>;
}

class InMemoryEventStore implements EventStore {
  private readonly streams = new Map<string, EventEnvelope[]>();
  private counter = 0;

  async read(streamId: string): Promise<EventEnvelope[]> {
    return [...(this.streams.get(streamId) ?? [])];
  }

  async append(
    streamId: string,
    events: InventoryEvent[],
    options: AppendOptions,
  ): Promise<EventEnvelope[]> {
    const current = this.streams.get(streamId) ?? [];
    if (current.length !== options.expectedVersion) {
      throw new ConcurrencyError(
        `stream ${streamId}: expected version ${options.expectedVersion}, actual ${current.length}`,
      );
    }

    const appended = events.map((payload, index): EventEnvelope => ({
      id: `evt-${++this.counter}`,
      streamId,
      sequence: current.length + index + 1,
      occurredAt: new Date().toISOString(),
      payload,
    }));

    this.streams.set(streamId, [...current, ...appended]);
    return appended;
  }
}

type InventoryCommand =
  | { type: "Receive"; quantity: number; reference: string }
  | { type: "Reserve"; quantity: number; orderId: string }
  | { type: "Release"; quantity: number; orderId: string }
  | { type: "Ship"; quantity: number; orderId: string }
  | { type: "Adjust"; delta: number; reason: string };

function decide(state: InventoryState, command: InventoryCommand): InventoryEvent[] {
  switch (command.type) {
    case "Receive":
      assertPositiveInteger(command.quantity, "quantity");
      if (!command.reference.trim()) throw new DomainError("reference is required");
      return [{
        type: "StockReceived",
        sku: state.sku,
        warehouse: state.warehouse,
        quantity: command.quantity,
        reference: command.reference,
      }];

    case "Reserve": {
      assertPositiveInteger(command.quantity, "quantity");
      if (!command.orderId.trim()) throw new DomainError("orderId is required");
      const available = state.onHand - state.reserved;
      if (command.quantity > available) {
        throw new DomainError(
          `insufficient available stock: requested=${command.quantity}, available=${available}`,
        );
      }
      return [{
        type: "StockReserved",
        sku: state.sku,
        warehouse: state.warehouse,
        quantity: command.quantity,
        orderId: command.orderId,
      }];
    }

    case "Release": {
      assertPositiveInteger(command.quantity, "quantity");
      const reservedForOrder = state.reservations.get(command.orderId) ?? 0;
      if (command.quantity > reservedForOrder) {
        throw new DomainError(
          `cannot release ${command.quantity}; order has ${reservedForOrder} reserved`,
        );
      }
      return [{
        type: "ReservationReleased",
        sku: state.sku,
        warehouse: state.warehouse,
        quantity: command.quantity,
        orderId: command.orderId,
      }];
    }

    case "Ship": {
      assertPositiveInteger(command.quantity, "quantity");
      const reservedForOrder = state.reservations.get(command.orderId) ?? 0;
      if (command.quantity > reservedForOrder) {
        throw new DomainError(
          `cannot ship ${command.quantity}; order has ${reservedForOrder} reserved`,
        );
      }
      return [{
        type: "StockShipped",
        sku: state.sku,
        warehouse: state.warehouse,
        quantity: command.quantity,
        orderId: command.orderId,
      }];
    }

    case "Adjust": {
      if (!Number.isInteger(command.delta) || command.delta === 0) {
        throw new DomainError("adjustment delta must be a non-zero integer");
      }
      if (!command.reason.trim()) throw new DomainError("adjustment reason is required");
      const resultingOnHand = state.onHand + command.delta;
      if (resultingOnHand < state.reserved) {
        throw new DomainError(
          `adjustment would reduce on-hand below reserved quantity (${state.reserved})`,
        );
      }
      return [{
        type: "StockAdjusted",
        sku: state.sku,
        warehouse: state.warehouse,
        delta: command.delta,
        reason: command.reason,
      }];
    }
  }
}

function streamId(sku: Sku, warehouse: WarehouseId): string {
  if (!sku.trim() || !warehouse.trim()) throw new Error("sku and warehouse are required");
  return `inventory:${warehouse}:${sku}`;
}

class InventoryService {
  constructor(private readonly store: EventStore) {}

  async execute(
    sku: Sku,
    warehouse: WarehouseId,
    command: InventoryCommand,
  ): Promise<InventoryState> {
    const id = streamId(sku, warehouse);
    const history = await this.store.read(id);
    const state = rebuild(sku, warehouse, history);
    const proposed = decide(state, command);

    const appended = await this.store.append(id, proposed, {
      expectedVersion: history.length,
    });

    return appended.reduce(
      (s, envelope) => apply(s, envelope.payload),
      state,
    );
  }

  async get(sku: Sku, warehouse: WarehouseId): Promise<InventoryState> {
    const id = streamId(sku, warehouse);
    const history = await this.store.read(id);
    return rebuild(sku, warehouse, history);
  }
}

// Projection used by read APIs. It intentionally derives values from aggregate
// state rather than exposing the internal reservation map directly.
interface InventoryView {
  sku: string;
  warehouse: string;
  onHand: number;
  reserved: number;
  available: number;
  shippedLifetime: number;
  version: number;
}

function toView(state: InventoryState): InventoryView {
  return {
    sku: state.sku,
    warehouse: state.warehouse,
    onHand: state.onHand,
    reserved: state.reserved,
    available: state.onHand - state.reserved,
    shippedLifetime: state.shipped,
    version: state.version,
  };
}

async function demo(): Promise<void> {
  const store = new InMemoryEventStore();
  const service = new InventoryService(store);

  await service.execute("FILTER-42", "TOR-1", {
    type: "Receive",
    quantity: 100,
    reference: "PO-2026-8841",
  });

  await service.execute("FILTER-42", "TOR-1", {
    type: "Reserve",
    quantity: 12,
    orderId: "SO-1007",
  });

  await service.execute("FILTER-42", "TOR-1", {
    type: "Ship",
    quantity: 8,
    orderId: "SO-1007",
  });

  await service.execute("FILTER-42", "TOR-1", {
    type: "Adjust",
    delta: -2,
    reason: "cycle count correction",
  });

  const finalState = await service.get("FILTER-42", "TOR-1");
  console.log(JSON.stringify(toView(finalState), null, 2));
}

void demo();
