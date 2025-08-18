FROM nginx:alpine

# Copy the static HTML file
COPY index.html /usr/share/nginx/html/index.html
COPY index.html /usr/share/nginx/html/behavioral.html

# Configure nginx to handle SPA routing
RUN echo 'server { \
    listen 80; \
    server_name localhost; \
    root /usr/share/nginx/html; \
    index index.html; \
    location / { \
        try_files $uri $uri/ /index.html; \
    } \
    location /behavioral { \
        try_files /index.html /index.html; \
    } \
}' > /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]