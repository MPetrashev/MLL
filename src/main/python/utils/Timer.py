import logging
import sys
import time

log = logging.getLogger(__name__)


class Timer:
    def __init__(self, name=None, limit=sys.float_info.max):
        self.name = name
        self.limit = limit
        self.start = time.time()
        log.info('Loading Timer')

    def __enter__(self):
        pass

    def __exit__(self, exception_type, exception_value, traceback):
        pass

    def __del__(self):
        period = time.time() - self.start
        if period > self.limit:
            raise Exception('The execution "%s" has exceeded limit: %.3fs' % (self.name, period))
        if self.name:
            if period > 60:
                log.info( '%s elapsed: %dm %.3fs', self.name, int(period / 60), period % 60)
            else:
                log.info('%s elapsed: %.3fs', self.name, period)
        else:
            log.info('Elapsed: %.3fs', period)
