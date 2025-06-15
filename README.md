# Tools for Data Science Project-1

Project - 1 for TDS course, 2025 June Term. 

The Task was to create a virtual teaching assistant that scrapes the web and gives the most appropriate response regarding a students query.

[Project Details](https://tds.s-anand.net/#/project-tds-virtual-ta)

The ultimate goal after scraping the discourse forums and the course contents, was to create an API endpoint, that would return an answer and the best posts/topics relating to the question.

There was an additional detail, regarding the inclusion of images, which were to be optional, that is implemented.

Initially I used OpenAI official APIs, but after reading up on the course matierial. I have switched to [aipipe.org's](https://aipipe.org/) open AI bindings/proxy. They work exactly the same. I just had to provide new embeddings and give a new base url.

[Aipipe documentation](https://github.com/sanand0/aipipe?tab=readme-ov-file#api)

The project can be run with:

```bash 
python3 -m venv .venv
source ./.venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Only one API endpoint is available, which accepts POST requests only (change the port accordingly):
> localhost:8000/api/