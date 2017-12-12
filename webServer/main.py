from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
	return 'Work in progress'
@app.route('/test')
def cakes():
	return 'test'

if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0', port=80)

