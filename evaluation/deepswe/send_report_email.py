#!/usr/bin/env python3
"""Mail a Markdown report to yourself over Gmail SMTP.

    GMAIL_APP_PASSWORD=… python3 send_report_email.py --to you@gmail.com --subject "…" report.md

The password is a Gmail *app password* (Google account → Security → App passwords),
read from the environment only; the sender defaults to the recipient. The report goes
in the body as plain text and again as a `.md` attachment so it survives mail clients
that reflow text.
"""

from __future__ import annotations

import argparse
import os
import smtplib
import ssl
import sys
from email.message import EmailMessage
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("report", type=Path)
    parser.add_argument("--to", required=True)
    parser.add_argument("--sender", default="", help="defaults to --to")
    parser.add_argument("--subject", default="DeepSWE run report")
    parser.add_argument("--attach", action="append", type=Path, default=[], help="extra files to attach (repeatable)")
    parser.add_argument("--smtp-host", default="smtp.gmail.com")
    parser.add_argument("--smtp-port", type=int, default=465)
    args = parser.parse_args(argv)

    password = os.environ.get("GMAIL_APP_PASSWORD", "").strip()
    if not password:
        print("GMAIL_APP_PASSWORD is not set", file=sys.stderr)
        return 2
    sender = args.sender or args.to
    body = args.report.read_text(encoding="utf-8")

    message = EmailMessage()
    message["From"] = sender
    message["To"] = args.to
    message["Subject"] = args.subject
    message.set_content(body)
    message.add_attachment(body.encode("utf-8"), maintype="text", subtype="markdown", filename=args.report.name)
    for extra in args.attach:
        message.add_attachment(extra.read_bytes(), maintype="application", subtype="octet-stream", filename=extra.name)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(args.smtp_host, args.smtp_port, context=context, timeout=60) as smtp:
        smtp.login(sender, password)
        smtp.send_message(message)
    print(f"sent {args.report} ({len(body)} chars) to {args.to}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
