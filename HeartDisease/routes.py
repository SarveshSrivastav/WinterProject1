from flask import Flask, render_template
from HeartDisease import app

#Routes
@app.route('/')
@app.route('/home')
def home():
  return render_template("home.html")

@app.route('/clips')
def clips():
  return render_template("clips.html")

@app.route('/sarvesh')
def sarvesh():
  return render_template("sarvesh.html")

@app.route('/yash')
def yash():
  return render_template("yash.html")

@app.route('/heart')
def heart():
  return render_template("heart.html")

@app.route('/graphs')
def graphs():
    return render_template("graphs.html")