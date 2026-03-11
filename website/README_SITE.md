Website demo

To use the demo locally:
1. Start the Flask server (from the repo root):
   - $env:FLASK_APP = 'src.server'
   - flask run --host=0.0.0.0 --port=5000
2. Serve the website directory:
   - python -m http.server 8000 --directory website
3. Open http://localhost:8000/index.html

Note: demo.js posts to /predict relative to the page origin. If you host the website separately, change the fetch URL to point to the Flask server's address.
