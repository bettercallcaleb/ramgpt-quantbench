/**
 * Asynchronous scheduler with:
 * - bounded concurrency
 * - token-bucket rate limiting
 * - per-job retry policy
 * - cancellation
 * - fair FIFO admission
 */

type JobId = string;

interface Job<T> {
  id: JobId;
  run(signal: AbortSignal): Promise<T>;
  maxAttempts?: number;
  retryBaseMs?: number;
}

interface SchedulerOptions {
  maxConcurrency: number;
  ratePerSecond: number;
  burst: number;
}

interface QueueEntry<T> {
  job: Job<T>;
  resolve(value: T): void;
  reject(reason: unknown): void;
  controller: AbortController;
  attempts: number;
  enqueuedAt: number;
}

class CancelledError extends Error {
  constructor(message = "job cancelled") {
    super(message);
    this.name = "CancelledError";
  }
}

class TokenBucket {
  private tokens: number;
  private lastRefill: number;

  constructor(
    private readonly ratePerSecond: number,
    private readonly capacity: number,
    private readonly now: () => number = () => performance.now(),
  ) {
    if (!(ratePerSecond > 0)) throw new Error("ratePerSecond must be > 0");
    if (!(capacity >= 1)) throw new Error("capacity must be >= 1");
    this.tokens = capacity;
    this.lastRefill = this.now();
  }

  private refill(): void {
    const current = this.now();
    const elapsedSeconds = Math.max(0, current - this.lastRefill) / 1000;
    this.lastRefill = current;
    this.tokens = Math.min(
      this.capacity,
      this.tokens + elapsedSeconds * this.ratePerSecond,
    );
  }

  tryTake(): boolean {
    this.refill();
    if (this.tokens < 1) return false;
    this.tokens -= 1;
    return true;
  }

  delayUntilTokenMs(): number {
    this.refill();
    if (this.tokens >= 1) return 0;
    const missing = 1 - this.tokens;
    return Math.ceil((missing / this.ratePerSecond) * 1000);
  }
}

export class RateLimitedScheduler {
  private readonly queue: QueueEntry<unknown>[] = [];
  private readonly active = new Map<JobId, QueueEntry<unknown>>();
  private readonly bucket: TokenBucket;
  private wakeTimer: ReturnType<typeof setTimeout> | null = null;
  private closed = false;

  constructor(private readonly options: SchedulerOptions) {
    if (!Number.isInteger(options.maxConcurrency) || options.maxConcurrency < 1) {
      throw new Error("maxConcurrency must be a positive integer");
    }
    this.bucket = new TokenBucket(options.ratePerSecond, options.burst);
  }

  schedule<T>(job: Job<T>): Promise<T> {
    if (this.closed) return Promise.reject(new Error("scheduler is closed"));
    if (!job.id) return Promise.reject(new Error("job id is required"));

    const duplicate =
      this.active.has(job.id) ||
      this.queue.some(entry => entry.job.id === job.id);

    if (duplicate) {
      return Promise.reject(new Error(`job id already exists: ${job.id}`));
    }

    return new Promise<T>((resolve, reject) => {
      const entry: QueueEntry<T> = {
        job,
        resolve,
        reject,
        controller: new AbortController(),
        attempts: 0,
        enqueuedAt: Date.now(),
      };
      this.queue.push(entry as QueueEntry<unknown>);
      this.pump();
    });
  }

  cancel(jobId: JobId): boolean {
    const queuedIndex = this.queue.findIndex(entry => entry.job.id === jobId);
    if (queuedIndex >= 0) {
      const [entry] = this.queue.splice(queuedIndex, 1);
      entry.controller.abort();
      entry.reject(new CancelledError(`queued job cancelled: ${jobId}`));
      return true;
    }

    const running = this.active.get(jobId);
    if (running) {
      running.controller.abort();
      return true;
    }

    return false;
  }

  close({ cancelPending = false } = {}): void {
    this.closed = true;
    if (cancelPending) {
      while (this.queue.length) {
        const entry = this.queue.shift()!;
        entry.controller.abort();
        entry.reject(new CancelledError("scheduler closed"));
      }
    }
    if (this.wakeTimer) {
      clearTimeout(this.wakeTimer);
      this.wakeTimer = null;
    }
  }

  stats() {
    return {
      queued: this.queue.length,
      active: this.active.size,
      closed: this.closed,
      oldestQueuedMs:
        this.queue.length > 0 ? Date.now() - this.queue[0].enqueuedAt : 0,
    };
  }

  private scheduleWake(delayMs: number): void {
    if (this.wakeTimer !== null) return;
    this.wakeTimer = setTimeout(() => {
      this.wakeTimer = null;
      this.pump();
    }, Math.max(1, delayMs));
  }

  private pump(): void {
    if (this.closed && this.queue.length === 0) return;

    while (
      this.active.size < this.options.maxConcurrency &&
      this.queue.length > 0
    ) {
      if (!this.bucket.tryTake()) {
        this.scheduleWake(this.bucket.delayUntilTokenMs());
        return;
      }

      const entry = this.queue.shift()!;
      if (entry.controller.signal.aborted) {
        entry.reject(new CancelledError());
        continue;
      }

      this.active.set(entry.job.id, entry);
      void this.execute(entry);
    }
  }

  private async execute(entry: QueueEntry<unknown>): Promise<void> {
    entry.attempts += 1;

    try {
      const value = await entry.job.run(entry.controller.signal);
      if (entry.controller.signal.aborted) {
        entry.reject(new CancelledError(`job aborted: ${entry.job.id}`));
      } else {
        entry.resolve(value);
      }
      this.active.delete(entry.job.id);
      this.pump();
      return;
    } catch (error) {
      this.active.delete(entry.job.id);

      if (entry.controller.signal.aborted) {
        entry.reject(new CancelledError(`job aborted: ${entry.job.id}`));
        this.pump();
        return;
      }

      const maxAttempts = entry.job.maxAttempts ?? 1;
      if (entry.attempts >= maxAttempts) {
        entry.reject(error);
        this.pump();
        return;
      }

      const base = entry.job.retryBaseMs ?? 250;
      const exponent = Math.min(entry.attempts - 1, 8);
      const deterministicBackoff = base * 2 ** exponent;
      const jitter = Math.floor(Math.random() * base);
      const delay = deterministicBackoff + jitter;

      setTimeout(() => {
        if (entry.controller.signal.aborted) {
          entry.reject(new CancelledError(`job aborted during retry wait: ${entry.job.id}`));
          this.pump();
          return;
        }
        entry.enqueuedAt = Date.now();
        this.queue.push(entry);
        this.pump();
      }, delay);

      this.pump();
    }
  }
}

// Demonstration: simulated API calls with both retry and cancellation behavior.
async function demo(): Promise<void> {
  const scheduler = new RateLimitedScheduler({
    maxConcurrency: 3,
    ratePerSecond: 4,
    burst: 2,
  });

  const results: Promise<string>[] = [];

  for (let i = 0; i < 10; i++) {
    let failuresRemaining = i % 4 === 0 ? 1 : 0;

    results.push(
      scheduler.schedule<string>({
        id: `job-${i}`,
        maxAttempts: 3,
        retryBaseMs: 100,
        async run(signal) {
          await new Promise<void>((resolve, reject) => {
            const timer = setTimeout(resolve, 80 + i * 5);
            signal.addEventListener("abort", () => {
              clearTimeout(timer);
              reject(new CancelledError());
            }, { once: true });
          });

          if (failuresRemaining > 0) {
            failuresRemaining -= 1;
            throw new Error(`transient failure in job-${i}`);
          }

          return `completed job-${i}`;
        },
      }),
    );
  }

  setTimeout(() => scheduler.cancel("job-8"), 120);

  const settled = await Promise.allSettled(results);
  for (const [index, result] of settled.entries()) {
    if (result.status === "fulfilled") {
      console.log(index, result.value);
    } else {
      console.log(index, "ERROR", String(result.reason));
    }
  }

  console.log("final stats", scheduler.stats());
  scheduler.close();
}

void demo();
