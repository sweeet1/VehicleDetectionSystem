"""
Shared brand icon drawing for the native window chrome and in-app title bar.
"""

from PyQt5.QtCore import QRectF, Qt
from PyQt5.QtGui import QColor, QIcon, QPainter, QPen, QPixmap

from ui.stylesheet import COLORS


def draw_kinpaku_mark(painter: QPainter, size: int):
    """Draw the gold control-room mark into a square drawing area."""
    scale = size / 34.0

    def s(value: float) -> float:
        return value * scale

    def p(value: float) -> int:
        return int(round(s(value)))

    painter.save()
    painter.setRenderHint(QPainter.Antialiasing)

    tile = QRectF(s(4.0), s(4.0), s(26.0), s(26.0))
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(COLORS["primary"]))
    painter.drawRoundedRect(tile, s(4.0), s(4.0))

    painter.setPen(QPen(QColor(COLORS["accent"]), max(1.0, s(3.0)), Qt.SolidLine, Qt.RoundCap))
    painter.drawLine(p(9), p(27), p(26), p(8))

    painter.setPen(QPen(QColor(COLORS["bg_surface"]), max(1.0, s(1.4)), Qt.SolidLine, Qt.RoundCap))
    painter.drawLine(p(13), p(27), p(28), p(12))

    painter.setPen(QPen(QColor(COLORS["success"]), max(1.0, s(1.2))))
    painter.setBrush(QColor(COLORS["success"]))
    painter.drawEllipse(QRectF(s(6.5), s(6.5), s(5.5), s(5.5)))

    painter.setPen(QPen(QColor(COLORS["accent"]), max(1.0, s(1.1))))
    painter.setBrush(Qt.NoBrush)
    painter.drawLine(p(7), p(22), p(13), p(22))
    painter.drawLine(p(21), p(8), p(27), p(8))

    painter.restore()


def create_app_icon() -> QIcon:
    """Build a multi-size QIcon so Windows title bars and taskbar stay crisp."""
    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        draw_kinpaku_mark(painter, size)
        painter.end()

        icon.addPixmap(pixmap)
    return icon
