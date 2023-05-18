FROM python:3.9
ENV PYTHONDONTWRITEBYTECODE 1
WORKDIR /project
COPY Pipfile Pipfile.lock /project/
RUN pip install pipenv && pipenv install --system
COPY . /project/
