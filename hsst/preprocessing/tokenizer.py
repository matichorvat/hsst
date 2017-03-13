import sarge


class Tokenizer(object):

    def __init__(self, cmd, timeout=1.0):
        self.text_cmd = cmd
        self.timeout = timeout
        self.start()

    def start(self):
        self.feeder = sarge.Feeder()
        self.reader = sarge.Capture()
        self.p = sarge.Command(self.text_cmd, stdout=self.reader)
        self.p.run(input=self.feeder, async=True)

    def end(self):
        # self.p.commands[0].terminate()
        self.p.terminate()
        self.feeder.close()
        self.reader.close()

    def tokenize(self, line):
        line = line.strip() + '\n'
        self.feeder.feed(line)
        self.p.poll()
        output = self.reader.readline(timeout=self.timeout).strip()

        if self.p.returncode is not None:
            self.end()
            self.start()
            return None

        else:
            return output
