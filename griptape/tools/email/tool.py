from __future__ import annotations

import logging
import smtplib
from email.mime.text import MIMEText
from typing import Optional

import schema
from attrs import Factory, define, field
from schema import Literal, Schema

from griptape.artifacts import ErrorArtifact, InfoArtifact, ListArtifact
from griptape.loaders.email_loader import EmailLoader
from griptape.tools import BaseTool
from griptape.utils.decorators import activity


@define
class EmailTool(BaseTool):
    """Tool for working with email.

    Attributes:
        username: Username/email address used to send email via the SMTP protocol and retrieve email via the IMAP protocol.
            Example: bender@futurama.com.
        password: Password used to send email via the SMTP protocol and retrieve email via the IMAP protocol. If using gmail,
            this would be an App Password.
        email_max_retrieve_count: Used to limit the number of messages retrieved during any given activities.
        smtp_host: Hostname or url of the SMTP server. Example: smtp.gmail.com. Required when using the `send` activity.
        smtp_port: Port of the SMTP server. Example: 465. Required when using the `send` activity.
        smtp_use_ssl: Whether to use SSL when sending email via the SMTP protocol.
        smtp_user: Username/email address used to send email via the SMTP protocol. Overrides username for SMTP only. Required when using the `send` activity.
        smtp_password: Password to send email via the SMTP protocol. Overrides password for SMTP only. Required when using the `send` activity.
        imap_url: Hostname or url of the IMAP server. Example: imap.gmail.com. Required when using the `retrieve` activity.
        imap_user: Username/email address used to retrieve email via the IMAP protocol. Overrides username for IMAP only. Required when using the `retrieve` activity.
        imap_password: Password to retrieve email via the IMAP protocol.  Overrides password for IMAP only. Required when using the `retrieve` activity.
        mailboxes: Descriptions of mailboxes available for retrieving email via the IMAP protocol. Required when using the `retrieve` activity.
            Example: {'INBOX': 'default mailbox for incoming email', 'SENT': 'default mailbox for sent email'}
        email_loader: Instance of `EmailLoader`.
    """

    username: Optional[str] = field(default=None, kw_only=True)
    password: Optional[str] = field(default=None, kw_only=True)
    email_max_retrieve_count: Optional[int] = field(default=None, kw_only=True)
    smtp_host: Optional[str] = field(default=None, kw_only=True)
    smtp_port: Optional[int] = field(default=None, kw_only=True)
    smtp_use_ssl: bool = field(default=True, kw_only=True)
    smtp_user: Optional[str] = field(default=Factory(lambda self: self.username, takes_self=True), kw_only=True)
    smtp_password: Optional[str] = field(default=Factory(lambda self: self.password, takes_self=True), kw_only=True)
    imap_url: Optional[str] = field(default=None, kw_only=True)
    imap_user: Optional[str] = field(default=Factory(lambda self: self.username, takes_self=True), kw_only=True)
    imap_password: Optional[str] = field(default=Factory(lambda self: self.password, takes_self=True), kw_only=True)
    mailboxes: Optional[dict[str, Optional[str]]] = field(default=None, kw_only=True)
    email_loader: EmailLoader = field(
        default=Factory(
            lambda self: EmailLoader(imap_url=self.imap_url, username=self.imap_user, password=self.imap_password),
            takes_self=True,
        ),
        kw_only=True,
    )

    @activity(
        config={
            "description": "Can be used to retrieve emails."
            "{% if _self.mailboxes %} Available mailboxes: {{ _self.mailboxes }}{% endif %}",
            "schema": Schema(
                {
                    Literal("label", description="Label to retrieve emails from such as 'INBOX' or 'SENT'"): str,
                    schema.Optional(
                        Literal("key", description="Optional key for filtering such as 'FROM' or 'SUBJECT'"),
                    ): str,
                    schema.Optional(
                        Literal("search_criteria", description="Optional search criteria to filter emails by key"),
                    ): str,
                    schema.Optional(Literal("max_count", description="Optional max email count")): int,
                },
            ),
        },
    )
    def retrieve(self, params: dict) -> ListArtifact | ErrorArtifact:
        if self.mailboxes is None:
            return ErrorArtifact("mailboxes is required")

        values = params["values"]
        max_count = int(values["max_count"]) if values.get("max_count") is not None else self.email_max_retrieve_count

        return self.email_loader.load(
            EmailLoader.EmailQuery(
                label=values["label"],
                key=values.get("key"),
                search_criteria=values.get("search_criteria"),
                max_count=max_count,
            ),
        )

    @activity(
        config={
            "description": "Can be used to send emails",
            "schema": Schema(
                {
                    Literal("to", description="Recipient's email address"): str,
                    Literal("subject", description="Email subject"): str,
                    Literal("body", description="Email body"): str,
                },
            ),
        },
    )
    def send(self, params: dict) -> InfoArtifact | ErrorArtifact:
        values = params["values"]
        if self.smtp_user is None:
            return ErrorArtifact("smtp_user is required")
        if self.smtp_password is None:
            return ErrorArtifact("smtp_password is required")
        if self.smtp_host is None:
            return ErrorArtifact("smtp_host is required")
        if self.smtp_port is None:
            return ErrorArtifact("smtp_port is required")

        msg = MIMEText(values["body"])
        msg["Subject"] = values["subject"]
        msg["From"] = self.smtp_user
        msg["To"] = values["to"]

        try:
            with self._create_smtp_client(self.smtp_host, self.smtp_port) as client:
                client.login(self.smtp_user, self.smtp_password)
                client.sendmail(msg["From"], [msg["To"]], msg.as_string())
                return InfoArtifact("email was successfully sent")
        except Exception as e:
            logging.error(e)
            return ErrorArtifact(f"error sending email: {e}")

    def _create_smtp_client(self, smtp_host: str, smtp_port: int) -> smtplib.SMTP | smtplib.SMTP_SSL:
        if self.smtp_use_ssl:
            return smtplib.SMTP_SSL(smtp_host, smtp_port)
        else:
            return smtplib.SMTP(smtp_host, smtp_port)
