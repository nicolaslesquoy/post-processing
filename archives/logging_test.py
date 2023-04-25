import logging

logging.basicConfig(filename='app.log',filemode="w",format="%(asctime)s | %(levelname)s | %(message)s",datefmt="%d-%b-%y %H:%M:%S")
logging.warning('truc')