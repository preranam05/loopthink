# Example data: an IT helpdesk

A small made-up dataset for trying the custom-data path. It is not from any real organisation.

- `requests.csv`: 200 requests across 8 intents, 25 each (`text`, `intent`).
- `unknown.csv`: 36 requests a helpdesk router should decline (facilities, HR, general questions).

Every request was written by hand as a distinct sentence, so the test split does not contain reworded
copies of training rows. It is still far cleaner than real tickets; treat the scores as a check that the
pipeline works, not as a forecast for real traffic.

    python scripts/train_custom.py --data examples/helpdesk/requests.csv --unknown examples/helpdesk/unknown.csv --out runs/helpdesk
