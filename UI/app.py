from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Your Python function
def my_python_function(search_term):
    # Replace with your actual function logic
    return f"Results for search term: {search_term}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    print("EWRTEWt")
    search_term = request.form['search_term']
    result = my_python_function(search_term)
    return jsonify(result=result)

if __name__ == '__main__':
    app.run(debug=True)
