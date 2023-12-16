import { defineConfig, sharpImageService } from "astro/config";
import react from '@astrojs/react';
import tailwind from "@astrojs/tailwind";
import mdx from "@astrojs/mdx";
import sitemap from "@astrojs/sitemap";

import vercel from "@astrojs/vercel/serverless";

// https://astro.build/config
export default defineConfig({
  output: 'hybrid',
  i18n: {
    defaultLocale: "es",
    locales: ["en", "es"],
    routing: {
      prefixDefaultLocale: true
    }
  },
  site: "https://codemils.com",
  image: {
    service: sharpImageService(),
    domains: ["source.unsplash.com", "images.unsplash.com", "dummyimage.com"]
  },
  integrations: [react({
    include: ['**/react/*']
  }), tailwind(), mdx(), sitemap()],
  adapter: vercel()
});