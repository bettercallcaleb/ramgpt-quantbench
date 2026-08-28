/**
 * Typed configuration merge engine.
 *
 * Supports layered configuration, schema-aware merge policies, deletion
 * markers, provenance reporting, and validation. Objects merge recursively,
 * arrays may replace or append according to schema, and unknown keys can be
 * rejected.
 */

type Primitive = string | number | boolean | null;
type ConfigValue = Primitive | ConfigObject | ConfigValue[];
interface ConfigObject { [key: string]: ConfigValue; }

type MergePolicy = "replace" | "deep" | "append" | "unique-append";

interface SchemaNode {
  type: "object" | "array" | "string" | "number" | "boolean" | "null";
  policy?: MergePolicy;
  required?: boolean;
  children?: Record<string, SchemaNode>;
  element?: SchemaNode;
  allowUnknown?: boolean;
}

interface Layer {
  name: string;
  value: ConfigObject;
}

interface ProvenanceEntry {
  path: string;
  layer: string;
  action: "set" | "replace" | "append" | "delete";
}

interface MergeResult {
  value: ConfigObject;
  provenance: ProvenanceEntry[];
}

const DELETE = "__DELETE__";

function isPlainObject(value: unknown): value is ConfigObject {
  return typeof value === "object" &&
    value !== null &&
    !Array.isArray(value);
}

function clone<T extends ConfigValue>(value: T): T {
  if (Array.isArray(value)) {
    return value.map(v => clone(v)) as T;
  }
  if (isPlainObject(value)) {
    return Object.fromEntries(
      Object.entries(value).map(([k, v]) => [k, clone(v)]),
    ) as T;
  }
  return value;
}

function pathJoin(base: string, key: string): string {
  return base ? `${base}.${key}` : key;
}

function stableKey(value: ConfigValue): string {
  if (Array.isArray(value)) {
    return `[${value.map(stableKey).join(",")}]`;
  }
  if (isPlainObject(value)) {
    return `{${Object.keys(value).sort().map(
      key => `${JSON.stringify(key)}:${stableKey(value[key])}`,
    ).join(",")}}`;
  }
  return JSON.stringify(value);
}

function actualType(value: ConfigValue): SchemaNode["type"] {
  if (value === null) return "null";
  if (Array.isArray(value)) return "array";
  if (isPlainObject(value)) return "object";
  return typeof value as "string" | "number" | "boolean";
}

function validateValue(
  value: ConfigValue,
  schema: SchemaNode,
  path = "",
  errors: string[] = [],
): string[] {
  const type = actualType(value);
  if (type !== schema.type) {
    errors.push(`${path || "<root>"}: expected ${schema.type}, got ${type}`);
    return errors;
  }

  if (schema.type === "object" && isPlainObject(value)) {
    const children = schema.children ?? {};

    for (const [key, childSchema] of Object.entries(children)) {
      if (childSchema.required && !(key in value)) {
        errors.push(`${pathJoin(path, key)}: required key is missing`);
      }
    }

    for (const [key, childValue] of Object.entries(value)) {
      const childSchema = children[key];
      if (!childSchema) {
        if (!schema.allowUnknown) {
          errors.push(`${pathJoin(path, key)}: unknown key`);
        }
        continue;
      }
      validateValue(childValue, childSchema, pathJoin(path, key), errors);
    }
  }

  if (schema.type === "array" && Array.isArray(value) && schema.element) {
    value.forEach((item, index) => {
      validateValue(item, schema.element!, `${path}[${index}]`, errors);
    });
  }

  return errors;
}

function mergeArrays(
  base: ConfigValue[],
  incoming: ConfigValue[],
  policy: MergePolicy,
): ConfigValue[] {
  if (policy === "replace") return clone(incoming);

  if (policy === "append") {
    return [...clone(base), ...clone(incoming)];
  }

  if (policy === "unique-append") {
    const result = clone(base);
    const seen = new Set(result.map(stableKey));
    for (const item of incoming) {
      const key = stableKey(item);
      if (!seen.has(key)) {
        result.push(clone(item));
        seen.add(key);
      }
    }
    return result;
  }

  throw new Error(`array does not support policy ${policy}`);
}

function mergeValue(
  base: ConfigValue | undefined,
  incoming: ConfigValue,
  schema: SchemaNode | undefined,
  layer: string,
  path: string,
  provenance: ProvenanceEntry[],
): ConfigValue | undefined {
  if (incoming === DELETE) {
    provenance.push({ path, layer, action: "delete" });
    return undefined;
  }

  if (schema && actualType(incoming) !== schema.type) {
    throw new Error(
      `${path}: layer ${layer} provides ${actualType(incoming)}, expected ${schema.type}`,
    );
  }

  if (Array.isArray(incoming)) {
    const policy = schema?.policy ?? "replace";
    if (base === undefined) {
      provenance.push({ path, layer, action: "set" });
      return clone(incoming);
    }
    if (!Array.isArray(base)) {
      provenance.push({ path, layer, action: "replace" });
      return clone(incoming);
    }
    provenance.push({
      path,
      layer,
      action: policy === "replace" ? "replace" : "append",
    });
    return mergeArrays(base, incoming, policy);
  }

  if (isPlainObject(incoming)) {
    const policy = schema?.policy ?? "deep";
    if (policy === "replace" || base === undefined || !isPlainObject(base)) {
      provenance.push({
        path,
        layer,
        action: base === undefined ? "set" : "replace",
      });
      return clone(incoming);
    }

    const output: ConfigObject = clone(base);
    const children = schema?.children ?? {};

    for (const [key, incomingValue] of Object.entries(incoming)) {
      const childPath = pathJoin(path, key);
      const childSchema = children[key];

      if (!childSchema && schema && schema.allowUnknown === false) {
        throw new Error(`${childPath}: unknown configuration key in layer ${layer}`);
      }

      const merged = mergeValue(
        output[key],
        incomingValue,
        childSchema,
        layer,
        childPath,
        provenance,
      );

      if (merged === undefined) delete output[key];
      else output[key] = merged;
    }
    return output;
  }

  provenance.push({
    path,
    layer,
    action: base === undefined ? "set" : "replace",
  });
  return incoming;
}

export function mergeConfiguration(
  layers: Layer[],
  schema: SchemaNode,
): MergeResult {
  if (schema.type !== "object") {
    throw new Error("root schema must be object");
  }

  let result: ConfigObject = {};
  const provenance: ProvenanceEntry[] = [];

  for (const layer of layers) {
    if (!layer.name.trim()) throw new Error("layer name must not be blank");

    const merged = mergeValue(
      result,
      layer.value,
      schema,
      layer.name,
      "",
      provenance,
    );

    if (!merged || !isPlainObject(merged)) {
      throw new Error(`layer ${layer.name} replaced root with non-object`);
    }
    result = merged;
  }

  const validationErrors = validateValue(result, schema);
  if (validationErrors.length) {
    throw new Error(
      `merged configuration failed validation:\n${validationErrors.join("\n")}`,
    );
  }

  return { value: result, provenance };
}

export function provenanceFor(
  provenance: ProvenanceEntry[],
  path: string,
): ProvenanceEntry[] {
  return provenance.filter(entry =>
    entry.path === path || entry.path.startsWith(`${path}.`)
  );
}

export function explainLastWriter(
  provenance: ProvenanceEntry[],
  path: string,
): ProvenanceEntry | null {
  const entries = provenance.filter(entry => entry.path === path);
  return entries.length ? entries[entries.length - 1] : null;
}

// Example schema for a service process.
const schema: SchemaNode = {
  type: "object",
  policy: "deep",
  allowUnknown: false,
  children: {
    service: {
      type: "object",
      required: true,
      policy: "deep",
      allowUnknown: false,
      children: {
        name: { type: "string", required: true },
        port: { type: "number", required: true },
        tags: {
          type: "array",
          policy: "unique-append",
          element: { type: "string" },
        },
      },
    },
    logging: {
      type: "object",
      policy: "deep",
      allowUnknown: false,
      children: {
        level: { type: "string", required: true },
        sinks: {
          type: "array",
          policy: "unique-append",
          element: { type: "string" },
        },
      },
    },
    features: {
      type: "object",
      policy: "deep",
      allowUnknown: true,
      children: {},
    },
  },
};

const defaults: Layer = {
  name: "defaults",
  value: {
    service: {
      name: "catalog-api",
      port: 8080,
      tags: ["http", "catalog"],
    },
    logging: {
      level: "info",
      sinks: ["stdout"],
    },
    features: {
      previewSearch: false,
      compactResponses: false,
    },
  },
};

const environment: Layer = {
  name: "production",
  value: {
    service: {
      port: 8443,
      tags: ["tls"],
    },
    logging: {
      sinks: ["structured-file"],
    },
    features: {
      compactResponses: true,
    },
  },
};

const emergencyOverride: Layer = {
  name: "incident-override",
  value: {
    logging: {
      level: "debug",
    },
    features: {
      previewSearch: DELETE,
    },
  },
};

const example = mergeConfiguration(
  [defaults, environment, emergencyOverride],
  schema,
);

console.log(JSON.stringify(example.value, null, 2));
console.log("service.port source:", explainLastWriter(example.provenance, "service.port"));
console.log("logging changes:", provenanceFor(example.provenance, "logging"));
