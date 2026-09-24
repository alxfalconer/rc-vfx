// Vercel build: publish the single-file UI. In serverless mode the API is this same deployment,
// under /api (a Python function) plus /api/upload (Blob client-upload tokens).
import { mkdirSync, copyFileSync, writeFileSync } from "node:fs";

mkdirSync("dist", { recursive: true });
copyFileSync("src/static/index.html", "dist/index.html");
writeFileSync("dist/config.js", 'window.RCV_API = "/api";\nwindow.RCV_MODE = "serverless";\n');
console.log("dist/ ready · serverless · API /api");
