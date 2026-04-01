/**
 * TS-08 — Social Media Format Cropping
 * Tools: Sharp (image processing) + Bannerbear API (styled overlays)
 *
 * Called from Python Agent 3 via subprocess.
 * Input: JSON payload on stdin
 * Output: JSON result on stdout
 *
 * Crops category-winner photos to social platform dimensions
 * and optionally applies Bannerbear styled overlays (property name,
 * vibe badge, rating display).
 *
 * Format dimensions:
 *   1:1  (square)   — 1080×1080  Instagram feed, Facebook feed
 *   9:16 (vertical) — 1080×1920  TikTok, Instagram Reels, Stories
 *   16:9 (landscape)— 1920×1080  Facebook, YouTube
 *
 * Usage (called by Python):
 *   node crop_worker.js < payload.json
 */

const sharp = require('sharp');
const https = require('https');
const http  = require('http');
const fs    = require('fs');
const path  = require('path');
const os    = require('os');

const BANNERBEAR_API_KEY = process.env.BANNERBEAR_API_KEY || '';
const BANNERBEAR_API_BASE = 'https://api.bannerbear.com/v2';

const CROP_DIMENSIONS = {
  '1_1':  { width: 1080, height: 1080 },
  '9_16': { width: 1080, height: 1920 },
  '16_9': { width: 1920, height: 1080 },
};

/**
 * Main entry point — reads JSON from stdin, processes, writes JSON to stdout
 */
async function main() {
  let input = '';
  process.stdin.on('data', chunk => { input += chunk; });
  process.stdin.on('end', async () => {
    try {
      const payload = JSON.parse(input);
      const result = await processCrops(payload);
      process.stdout.write(JSON.stringify(result));
    } catch (err) {
      process.stdout.write(JSON.stringify({
        success: false,
        error: err.message,
        crops: [],
      }));
    }
  });
}

/**
 * Process all crop requests from the payload.
 *
 * Payload shape:
 * {
 *   property_id: string,
 *   vibe_profile: string,
 *   property_name: string,
 *   crops: [
 *     {
 *       source_url: string,
 *       formats: ["1_1", "9_16", "16_9"],
 *       category: string,
 *       apply_overlay: boolean,
 *       overlay_template_id: string | null,
 *     }
 *   ]
 * }
 */
async function processCrops(payload) {
  const { property_id, vibe_profile, property_name, crops } = payload;
  const results = [];

  for (const crop of crops) {
    try {
      const imageBuffer = await downloadImage(crop.source_url);
      if (!imageBuffer) {
        results.push({ source_url: crop.source_url, error: 'Download failed', crops: [] });
        continue;
      }

      const cropResults = [];
      for (const format of crop.formats) {
        const dims = CROP_DIMENSIONS[format];
        if (!dims) continue;

        // Sharp crop — smart crop centres on the most interesting region
        const cropped = await sharp(imageBuffer)
          .resize(dims.width, dims.height, {
            fit: 'cover',
            position: sharp.strategy.attention,  // Smart crop — focus on subject
          })
          .jpeg({ quality: 92, progressive: true })
          .toBuffer();

        let finalBuffer = cropped;

        // Bannerbear overlay (optional — only for formats that need it)
        if (crop.apply_overlay && crop.overlay_template_id && BANNERBEAR_API_KEY) {
          const overlaid = await applyBannerbearOverlay(
            cropped,
            crop.overlay_template_id,
            property_name,
            vibe_profile,
          );
          if (overlaid) finalBuffer = overlaid;
        }

        // Write to temp file — Python side will upload to R2
        const tmpFile = path.join(
          os.tmpdir(),
          `crop_${property_id}_${crop.category}_${format}_${Date.now()}.jpg`
        );
        fs.writeFileSync(tmpFile, finalBuffer);

        cropResults.push({
          format,
          category: crop.category,
          tmp_file: tmpFile,
          width: dims.width,
          height: dims.height,
          has_overlay: crop.apply_overlay && !!overlaid,
        });
      }

      results.push({
        source_url: crop.source_url,
        crops: cropResults,
        error: null,
      });

    } catch (err) {
      results.push({
        source_url: crop.source_url,
        error: err.message,
        crops: [],
      });
    }
  }

  return { success: true, property_id, results };
}

/**
 * Apply a Bannerbear overlay template to a cropped image buffer.
 * Returns the overlaid image buffer or null on failure.
 */
async function applyBannerbearOverlay(
  imageBuffer,
  templateUid,
  propertyName,
  vibeProfile,
) {
  if (!BANNERBEAR_API_KEY) return null;

  try {
    // Upload the base image to Bannerbear (it requires a URL, not bytes)
    // In production: upload to R2 first, pass the R2 URL to Bannerbear
    // Here we use Bannerbear's base64 input if available, or skip overlay
    const modifications = [
      {
        name: 'property_name',
        text: propertyName || '',
      },
      {
        name: 'vibe_badge',
        text: vibeProfile.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
      },
    ];

    const response = await bannerbearRequest('POST', '/images', {
      template: templateUid,
      modifications,
      synchronous: true,  // Wait for completion
    });

    if (response && response.image_url) {
      return await downloadImage(response.image_url);
    }
    return null;
  } catch (err) {
    console.error(`Bannerbear overlay failed: ${err.message}`);
    return null;
  }
}

/**
 * Make a Bannerbear API request.
 */
function bannerbearRequest(method, endpoint, body) {
  return new Promise((resolve, reject) => {
    const data = JSON.stringify(body);
    const options = {
      hostname: 'api.bannerbear.com',
      port: 443,
      path: `/v2${endpoint}`,
      method,
      headers: {
        'Authorization': `Bearer ${BANNERBEAR_API_KEY}`,
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(data),
      },
    };

    const req = https.request(options, res => {
      let responseData = '';
      res.on('data', chunk => { responseData += chunk; });
      res.on('end', () => {
        try {
          resolve(JSON.parse(responseData));
        } catch {
          resolve(null);
        }
      });
    });
    req.on('error', reject);
    req.write(data);
    req.end();
  });
}

/**
 * Download image from URL to buffer.
 * Handles both http and https.
 */
function downloadImage(url) {
  return new Promise((resolve, reject) => {
    const protocol = url.startsWith('https') ? https : http;
    protocol.get(url, { timeout: 30000 }, res => {
      if (res.statusCode === 301 || res.statusCode === 302) {
        // Follow redirect
        resolve(downloadImage(res.headers.location));
        return;
      }
      if (res.statusCode !== 200) {
        resolve(null);
        return;
      }
      const chunks = [];
      res.on('data', chunk => chunks.push(chunk));
      res.on('end', () => resolve(Buffer.concat(chunks)));
      res.on('error', () => resolve(null));
    }).on('error', () => resolve(null));
  });
}

main();
