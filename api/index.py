from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def hello():
    return '''
    <html>
    <head><title>Golf Prediction System</title></head>
    <body style="background-color: #212529; color: white; font-family: Arial; padding: 20px;">
        <h1>🏌️ Golf Prediction System</h1>
        <p>Status: Working on Vercel!</p>
        <p><a href="/api/health" style="color: #17a2b8;">Check API Health</a></p>
        <p><a href="/predictions" style="color: #17a2b8;">View Predictions</a></p>
    </body>
    </html>
    '''

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'healthy',
        'platform': 'vercel',
        'message': 'Golf prediction system is running!'
    })

@app.route('/predictions')
def predictions():
    return jsonify({
        'status': 'success',
        'predictions': [
            {'player': 'Scottie Scheffler', 'score': 0.95, 'rank': 1},
            {'player': 'Rory McIlroy', 'score': 0.88, 'rank': 2},
            {'player': 'Jon Rahm', 'score': 0.82, 'rank': 3}
        ],
        'count': 3
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
