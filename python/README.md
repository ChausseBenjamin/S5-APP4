# UV Setup

Each file inside of `src` is both a library **and** a script.
The reasoning behind it is that when testing specific functionality, a single
file can be run. But the code developped in one file can the be built upon in
later files without removing (or having to run) prior tests.

For example, I can test that `src/img.py` works as expected by just running

```py
uv run src/img.py
```

But later, I can re-use the code from `img.py` by simply importing it and
calling its functions.

```py
import src.img

def my_func():
    data = img.original()
    # ...
```
