"""Write a file only when its contents would change."""
import os


def write_if_changed(path, text, encoding="utf-8"):
    if os.path.exists(path):
        try:
            if open(path, encoding=encoding).read() == text:
                return False
        except (UnicodeDecodeError, OSError):
            pass
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding=encoding) as fh:
        fh.write(text)
    return True


def frame_if_changed(path, df, **kw):
    import io
    buf = io.StringIO()
    df.to_csv(buf, **kw)
    return write_if_changed(path, buf.getvalue())
