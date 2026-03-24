def app(environ, start_response):
    start_response("200 OK", [("Content-Type", "application/json")])
    return [b'{"success": true, "message": "SSV API Root"}']
