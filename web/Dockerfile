# Build estático + nginx. El cómputo ocurre en el navegador (WebGPU), así que
# el contenedor NO necesita GPU, ni CUDA, ni acceso a /dev/dri.
FROM node:22-alpine AS build
WORKDIR /app
COPY package*.json .npmrc ./
RUN npm ci --no-fund --no-audit
COPY . .
RUN npm run build

FROM nginx:alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD wget -qO- http://127.0.0.1/ >/dev/null || exit 1
