// Vercel Node function: issues client-upload tokens so the browser can send clips (up to 200 MB)
// straight to Vercel Blob, around the 4.5 MB function body limit. Needs BLOB_READ_WRITE_TOKEN,
// which Vercel adds when a Blob store is connected to the project.
import { handleUpload } from "@vercel/blob/client";

const VIDEO = ["video/mp4", "video/quicktime", "video/webm", "video/x-matroska", "video/x-m4v", "video/x-msvideo", "video/*"];

export default async function handler(request) {
  if (request.method !== "POST") return new Response("POST only", { status: 405 });
  try {
    const body = await request.json();
    const json = await handleUpload({
      body,
      request,
      onBeforeGenerateToken: async (pathname) => {
        // prototype: no accounts, so constrain what an anonymous token can do
        if (!/^clips\/[\w.\- ]{1,120}$/.test(pathname)) throw new Error("uploads go under clips/");
        return {
          allowedContentTypes: VIDEO,
          maximumSizeInBytes: 200 * 1024 * 1024,
          addRandomSuffix: true,
        };
      },
      onUploadCompleted: async () => {},
    });
    return Response.json(json);
  } catch (error) {
    return Response.json({ error: error.message }, { status: 400 });
  }
}
