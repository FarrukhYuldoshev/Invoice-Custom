## Welcome to
# INVOICE READER


### About the Project

soon...

### About the BackEnd

soon...

### About the FrontEnd
You can view project in github with link: 


***

## Tech

* [FastAPI](https://fastapi.tiangolo.com/) - is a high-level `Python Web framework`
* [PostgreSQL](https://www.postgresql.org/) - open source object-relational database system

And many other libraries.

Dillinger requires [Python](https://www.python.org) `v3.11.0 > 3.11.6 <3.12`.

```shell
$ git clone https://github.com/FarrukhYuldoshev/Invoice-Custom
```

***

## Setting project

* `Linux`
```shell
$ virtualenv -p /usr/bin/python3 .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

* `Windows`
```shell
$ python -m venv ./venv
$ venv\Scripts\activate
$ pip install -r requirements.txt
```

* `MacBook`
```shell
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

***

## Development
### Configure `PostgreSQL`
Create clear database named `invoice_db`.

```shell
$ sudo -u postgres psql
postgres=# ...
CREATE DATABASE invoice_db;
CREATE USER invoice_admin WITH PASSWORD 'invoice_1234';
ALTER ROLE invoice_admin SET client_encoding TO 'utf8';
ALTER ROLE invoice_admin SET timezone TO 'UTC';
GRANT ALL PRIVILEGES ON DATABASE invoice_db TO invoice_admin;
GRANT ALL PRIVILEGES ON SCHEMA public TO invoice_admin;
\q
$ sudo -u postgres psql -d invoice_db
GRANT USAGE ON SCHEMA public TO invoice_admin;
GRANT CREATE ON SCHEMA public TO invoice_admin;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO invoice_admin;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO invoice_admin;

```
Migrate to database and run project.
```shell
$ poetry install --sync
$ poetry -lock --no-update
$ alembic upgrade head
python main.py
```
`Output`
```shell
INFO:     Will watch for changes in these directories: ['/current path/']
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [28389] using StatReload
INFO:     Started server process [28391]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```
Open http://127.0.0.1:8000 in your browser for see result.