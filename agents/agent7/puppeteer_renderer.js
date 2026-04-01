/**
 * TS-23 — Puppeteer PDF Renderer
 * Tool: Puppeteer (headless Chrome)
 *
 * Takes an HTML file path and PDF output path as CLI arguments.
 * Renders the HTML in headless Chrome and prints to PDF.
 * Called via subprocess from report_generator.py.
 *
 * Usage: node puppeteer_renderer.js <html_path> <pdf_path>
 *
 * Why Puppeteer over wkhtmltopdf or jsPDF:
 *   - Full Chrome rendering engine = pixel-perfect CSS fidelity
 *   - Modern CSS grid, flexbox, custom fonts all supported
 *   - The report HTML uses modern CSS — wkhtmltopdf degrades it
 *   - Production standard for PDF generation in Node.js SaaS
 *
 * Runs as a scheduled batch job (last day of each month), not real-time.
 * Compute cost absorbed in LangGraph service infrastructure.
 */

const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');

async function renderPdf(htmlPath, pdfPath) {
  // Validate inputs
  if (!htmlPath || !pdfPath) {
    console.error('Usage: node puppeteer_renderer.js <html_path> <pdf_path>');
    process.exit(1);
  }

  if (!fs.existsSync(htmlPath)) {
    console.error(`HTML file not found: ${htmlPath}`);
    process.exit(1);
  }

  const browser = await puppeteer.launch({
    headless: 'new',
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-dev-shm-usage',   // Required for Docker/Railway environments
      '--disable-gpu',
    ],
  });

  try {
    const page = await browser.newPage();

    // Load the HTML file using file:// protocol for local files
    const fileUrl = `file://${path.resolve(htmlPath)}`;
    await page.goto(fileUrl, {
      waitUntil: 'networkidle0',  // Wait for fonts and images to load
      timeout: 30000,
    });

    // Print to PDF — Letter size, margins matching the report design
    await page.pdf({
      path: pdfPath,
      format: 'Letter',
      margin: {
        top: '0',
        right: '0',
        bottom: '0',
        left: '0',
      },
      printBackground: true,   // Include background colors
      preferCSSPageSize: false,
    });

    console.log(`PDF rendered successfully: ${pdfPath}`);

  } finally {
    await browser.close();
  }
}

// Entry point
const [,, htmlPath, pdfPath] = process.argv;

renderPdf(htmlPath, pdfPath)
  .then(() => process.exit(0))
  .catch(err => {
    console.error('Puppeteer render failed:', err.message);
    process.exit(1);
  });
