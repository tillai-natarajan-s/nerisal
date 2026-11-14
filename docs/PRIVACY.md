# Privacy and responsible use

NERISAL measures how crowded a space is. It is designed so that it never needs to know who anyone is.

## What NERISAL does not do

- No face recognition, no identity, no tracking of individuals across cameras or days.
- No video leaves the device. Nothing is sent to the internet or a cloud service.
- No video is recorded continuously.

## What it does with images

| Data | Where it goes | Protection |
|---|---|---|
| Live camera frames | processed in memory on the Pi | every detected head is blurred and density-counted regions are pixelated before a frame is shown or saved |
| Dashboard video | local network only, from the Pi to screens on site | already blurred |
| Snapshots | saved on the Pi only when staff press *Save snapshot* | taken from the blurred frame |
| Incident snapshots | saved on the Pi when the grade reaches E or F (at most four per incident) | taken from the blurred frame |
| Logs | one line per second: counts, densities, grades | numbers only, no images |
| Uploaded test files | kept on the Pi, only the last five | deleted automatically after that |

## Why on the device

Processing on the device keeps personal data to a minimum and keeps working where networks are jammed, which is
common in large crowds. This follows the data-minimisation principle of India's Digital Personal Data Protection
Act, 2023: only the numbers needed for safety are produced, and images are not shared.

## Responsible deployment

- Put up signs saying crowd density is being measured, with no identification.
- Give access to the dashboard only to safety staff.
- Delete snapshots and incident records after the event unless they are needed for a review.
- Use the grades to manage flow (open exits, hold entry), not to single out individuals.

## Data used to build the models

- Head detector: trained on CrowdHuman (CC BY-NC 4.0) for non-commercial research. No CrowdHuman images are
  published in this repository.
- Density model: pretrained DM-Count weights (MIT licence) from the authors.
- Our test photos come from Wikimedia Commons, Pexels and Unsplash; the sources are listed in
  `results/hand_counts_india_sources.txt` and the photos are not redistributed here.
