import sys

from PySide6.QtCore import Slot, Signal, QObject

class WorkerSignals(QObject):
	finished = Signal()
	error = Signal(tuple)
	progress = Signal(int)
	result = Signal(object)