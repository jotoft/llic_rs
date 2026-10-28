// Runs a wasm32-wasip1 program in Node (V8, the web demo's engine) with the
// current directory and the root preopened. Also works as a cargo test runner:
//   CARGO_TARGET_WASM32_WASIP1_RUNNER="node scripts/wasm_bench.mjs"
//
//   node scripts/wasm_bench.mjs program.wasm [args...]
import { readFile } from "node:fs/promises";
import { WASI } from "node:wasi";
import process from "node:process";

const [wasmPath, ...args] = process.argv.slice(2);
if (!wasmPath) {
  console.error("usage: node scripts/wasm_bench.mjs program.wasm [args...]");
  process.exit(2);
}
const wasi = new WASI({
  version: "preview1",
  args: [wasmPath, ...args],
  // "." for relative image paths; "/" for tests that use absolute paths
  // (CARGO_MANIFEST_DIR).
  preopens: { ".": process.cwd(), "/": "/" },
  returnOnExit: true,
});
const module = await WebAssembly.compile(await readFile(wasmPath));
const instance = await WebAssembly.instantiate(module, wasi.getImportObject());
process.exitCode = wasi.start(instance);
