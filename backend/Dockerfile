FROM python:3.12-slim
WORKDIR /code
COPY ./backend/requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/backend
COPY --chown=user ./backend $HOME/backend
# CMD ["uvicorn", "api:app", "--port", "8000", "--loop", "asyncio"]