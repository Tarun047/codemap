# Code Map wiki

Welcome to codemap, this document holds the important information that helps you navigate througout the code map 
project.

Code Map is a privacy focused, first of it's kind code search engine that helps democratize the area of programming, by making code easily searchable to anyone.

## Inception
So far we've developed complex search engines, which can sift through Terrabytes if not Petabytes of data and give us the most relavant information quickly.

However we feel that the search hasn't evolved much in the last couple of decades when it comes to codebases. 

Because when we search on codebases, we still use cryptic shortforms or syntaxes to naviagate through the code. 

Well, now we can put an end to this and democratize the area of code search as well.

Interact with your codebase as if it was another human.

## Contribute

1. To contribute to this repo, first clone the repo locally on your machine using `https://github.com/Tarun047/codemap.git`

2. Install [ollama](https://ollama.com/download) and [podman desktop](https://podman-desktop.io/downloads)

3. Create a new python virtual environment using 
`python3 -m venv venv`

4. Install the requirements to the newly added virtual environment using `pip install -r requirements.txt`

4. Run the `chromadb` docker image locally on your machine using `podman run -d -p 8000:8000 chromadb/chroma `

5. Then launch the app using `python main.py`


