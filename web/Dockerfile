FROM node:18-alpine

WORKDIR /app

COPY package.json ./
COPY server.js ./
COPY index.html ./
COPY build.js ./

# Copy pre-packaged static results if present
RUN mkdir -p results
COPY results/ results/ 2>/dev/null || true

EXPOSE 3000

ENV PORT=3000
CMD ["node", "server.js"]
