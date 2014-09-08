formula = """${\mathrm{P[\;}}\tilde{\Omega}_{n,k}{\mathrm{\;]}}=1$"""


class Command:
	def __init__(self, name):
		self.name = name
		self.args = []

	def addArg(self, arg):
		self.args.append(arg)