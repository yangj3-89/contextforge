FROM node:22-alpine
WORKDIR /ui
COPY package.json package-lock.json* ./
RUN npm install --no-audit --no-fund
COPY . .
EXPOSE 5173
# VITE_API_URL is read at build time, so build when the container starts.
CMD ["sh", "-c", "npm run build && npm run preview -- --host 0.0.0.0 --port 5173"]
