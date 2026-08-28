/**
 * Offline form synchronization library.
 *
 * Changes are written to IndexedDB first, then replicated to an HTTP endpoint
 * when connectivity permits. Each form mutation carries a client-generated
 * operation id so retries are idempotent on the server.
 */

const DB_NAME = "offline-form-sync";
const DB_VERSION = 1;
const FORMS_STORE = "forms";
const OPS_STORE = "operations";

function openDatabase() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);

    req.onupgradeneeded = () => {
      const db = req.result;

      if (!db.objectStoreNames.contains(FORMS_STORE)) {
        const forms = db.createObjectStore(FORMS_STORE, { keyPath: "formId" });
        forms.createIndex("updatedAt", "updatedAt");
      }

      if (!db.objectStoreNames.contains(OPS_STORE)) {
        const ops = db.createObjectStore(OPS_STORE, { keyPath: "operationId" });
        ops.createIndex("state", "state");
        ops.createIndex("createdAt", "createdAt");
      }
    };

    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function requestAsPromise(request) {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

function transactionDone(tx) {
  return new Promise((resolve, reject) => {
    tx.oncomplete = resolve;
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error || new Error("transaction aborted"));
  });
}

function newOperationId() {
  if (crypto.randomUUID) return crypto.randomUUID();
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  return [...bytes].map(b => b.toString(16).padStart(2, "0")).join("");
}

function clone(value) {
  return structuredClone(value);
}

export class OfflineFormStore {
  constructor({ endpoint, fetchImpl = fetch, now = () => new Date().toISOString() }) {
    this.endpoint = endpoint.replace(/\/$/, "");
    this.fetchImpl = fetchImpl;
    this.now = now;
    this.dbPromise = openDatabase();
    this.syncing = null;
  }

  async saveDraft(formId, fields, baseVersion = null) {
    if (!formId) throw new Error("formId is required");

    const db = await this.dbPromise;
    const tx = db.transaction([FORMS_STORE, OPS_STORE], "readwrite");
    const forms = tx.objectStore(FORMS_STORE);
    const ops = tx.objectStore(OPS_STORE);

    const existing = await requestAsPromise(forms.get(formId));
    const updatedAt = this.now();
    const operationId = newOperationId();

    const record = {
      formId,
      fields: clone(fields),
      baseVersion: existing?.baseVersion ?? baseVersion,
      serverVersion: existing?.serverVersion ?? baseVersion,
      syncState: "pending",
      updatedAt,
      lastError: null,
    };

    forms.put(record);
    ops.put({
      operationId,
      formId,
      kind: "upsert",
      fields: clone(fields),
      baseVersion: record.serverVersion,
      state: "pending",
      attemptCount: 0,
      createdAt: updatedAt,
      nextAttemptAt: updatedAt,
      lastError: null,
    });

    await transactionDone(tx);
    return clone(record);
  }

  async deleteForm(formId) {
    const db = await this.dbPromise;
    const tx = db.transaction([FORMS_STORE, OPS_STORE], "readwrite");
    const forms = tx.objectStore(FORMS_STORE);
    const ops = tx.objectStore(OPS_STORE);
    const existing = await requestAsPromise(forms.get(formId));

    const operationId = newOperationId();
    const now = this.now();

    forms.put({
      formId,
      fields: existing?.fields ?? {},
      baseVersion: existing?.baseVersion ?? null,
      serverVersion: existing?.serverVersion ?? null,
      syncState: "pending-delete",
      updatedAt: now,
      lastError: null,
    });

    ops.put({
      operationId,
      formId,
      kind: "delete",
      fields: null,
      baseVersion: existing?.serverVersion ?? null,
      state: "pending",
      attemptCount: 0,
      createdAt: now,
      nextAttemptAt: now,
      lastError: null,
    });

    await transactionDone(tx);
  }

  async getForm(formId) {
    const db = await this.dbPromise;
    const tx = db.transaction(FORMS_STORE, "readonly");
    const result = await requestAsPromise(tx.objectStore(FORMS_STORE).get(formId));
    await transactionDone(tx);
    return result ? clone(result) : null;
  }

  async listForms() {
    const db = await this.dbPromise;
    const tx = db.transaction(FORMS_STORE, "readonly");
    const result = await requestAsPromise(tx.objectStore(FORMS_STORE).getAll());
    await transactionDone(tx);
    return result.map(clone);
  }

  async pendingOperations() {
    const db = await this.dbPromise;
    const tx = db.transaction(OPS_STORE, "readonly");
    const index = tx.objectStore(OPS_STORE).index("state");
    const pending = await requestAsPromise(index.getAll("pending"));
    await transactionDone(tx);

    const nowMs = Date.now();
    return pending
      .filter(op => Date.parse(op.nextAttemptAt) <= nowMs)
      .sort((a, b) => a.createdAt.localeCompare(b.createdAt));
  }

  backoff(attemptCount) {
    const base = Math.min(60_000, 1000 * 2 ** Math.min(attemptCount, 6));
    const jitter = Math.floor(Math.random() * Math.max(250, base * 0.2));
    return base + jitter;
  }

  async markOperationFailure(operation, error) {
    const db = await this.dbPromise;
    const tx = db.transaction([FORMS_STORE, OPS_STORE], "readwrite");
    const ops = tx.objectStore(OPS_STORE);
    const forms = tx.objectStore(FORMS_STORE);

    const stored = await requestAsPromise(ops.get(operation.operationId));
    if (!stored) return transactionDone(tx);

    const attemptCount = stored.attemptCount + 1;
    const delay = this.backoff(attemptCount);
    stored.attemptCount = attemptCount;
    stored.lastError = String(error);
    stored.nextAttemptAt = new Date(Date.now() + delay).toISOString();
    ops.put(stored);

    const form = await requestAsPromise(forms.get(operation.formId));
    if (form) {
      form.lastError = String(error);
      forms.put(form);
    }

    await transactionDone(tx);
  }

  async applyServerSuccess(operation, response) {
    const db = await this.dbPromise;
    const tx = db.transaction([FORMS_STORE, OPS_STORE], "readwrite");
    const forms = tx.objectStore(FORMS_STORE);
    const ops = tx.objectStore(OPS_STORE);

    if (operation.kind === "delete") {
      forms.delete(operation.formId);
    } else {
      const form = await requestAsPromise(forms.get(operation.formId));
      if (form) {
        form.serverVersion = response.version;
        form.baseVersion = response.version;
        form.syncState = "synced";
        form.lastError = null;
        if (response.fields) form.fields = clone(response.fields);
        forms.put(form);
      }
    }

    ops.delete(operation.operationId);
    await transactionDone(tx);
  }

  async recordConflict(operation, response) {
    const db = await this.dbPromise;
    const tx = db.transaction([FORMS_STORE, OPS_STORE], "readwrite");
    const forms = tx.objectStore(FORMS_STORE);
    const ops = tx.objectStore(OPS_STORE);

    const form = await requestAsPromise(forms.get(operation.formId));
    if (form) {
      form.syncState = "conflict";
      form.conflict = {
        localFields: clone(form.fields),
        serverFields: clone(response.fields),
        serverVersion: response.version,
        operationId: operation.operationId,
      };
      forms.put(form);
    }

    operation.state = "conflict";
    operation.lastError = "version conflict";
    ops.put(operation);
    await transactionDone(tx);
  }

  async sendOperation(operation) {
    const url = `${this.endpoint}/forms/${encodeURIComponent(operation.formId)}`;
    const response = await this.fetchImpl(url, {
      method: operation.kind === "delete" ? "DELETE" : "PUT",
      headers: {
        "content-type": "application/json",
        "x-operation-id": operation.operationId,
      },
      body: operation.kind === "delete"
        ? JSON.stringify({ baseVersion: operation.baseVersion })
        : JSON.stringify({
            fields: operation.fields,
            baseVersion: operation.baseVersion,
          }),
    });

    if (response.status === 409) {
      return { kind: "conflict", body: await response.json() };
    }
    if (!response.ok) {
      const text = await response.text();
      throw new Error(`sync failed ${response.status}: ${text.slice(0, 200)}`);
    }
    return { kind: "success", body: await response.json() };
  }

  async sync() {
    if (this.syncing) return this.syncing;

    this.syncing = (async () => {
      try {
        const operations = await this.pendingOperations();
        for (const operation of operations) {
          try {
            const result = await this.sendOperation(operation);
            if (result.kind === "conflict") {
              await this.recordConflict(operation, result.body);
            } else {
              await this.applyServerSuccess(operation, result.body);
            }
          } catch (error) {
            await this.markOperationFailure(operation, error);
          }
        }
      } finally {
        this.syncing = null;
      }
    })();

    return this.syncing;
  }

  async resolveConflict(formId, resolver) {
    const form = await this.getForm(formId);
    if (!form?.conflict) throw new Error(`form ${formId} has no conflict`);

    const merged = await resolver({
      local: clone(form.conflict.localFields),
      server: clone(form.conflict.serverFields),
    });

    const db = await this.dbPromise;
    const tx = db.transaction([FORMS_STORE, OPS_STORE], "readwrite");
    const forms = tx.objectStore(FORMS_STORE);
    const ops = tx.objectStore(OPS_STORE);

    const current = await requestAsPromise(forms.get(formId));
    current.fields = clone(merged);
    current.serverVersion = form.conflict.serverVersion;
    current.baseVersion = form.conflict.serverVersion;
    current.syncState = "pending";
    delete current.conflict;
    forms.put(current);

    const op = {
      operationId: newOperationId(),
      formId,
      kind: "upsert",
      fields: clone(merged),
      baseVersion: form.conflict.serverVersion,
      state: "pending",
      attemptCount: 0,
      createdAt: this.now(),
      nextAttemptAt: this.now(),
      lastError: null,
    };
    ops.put(op);

    await transactionDone(tx);
    return clone(current);
  }
}

// Example wiring:
export function installAutomaticSync(store) {
  const trigger = () => {
    if (navigator.onLine) void store.sync();
  };

  window.addEventListener("online", trigger);
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") trigger();
  });

  const timer = setInterval(trigger, 30_000);

  return () => {
    clearInterval(timer);
    window.removeEventListener("online", trigger);
  };
}
