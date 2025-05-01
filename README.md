# MediMatch

Develop a medication recommendation system that uses data specific to each individual patient and drug information to suggest the most effective treatments

# Team members:

|      Name      |
| :------------: |
|  Sydney Boles  |
|  Alexis Lewis  |
| Melody Giberson|

## Getting Started

Before running the project, ensure you have:

Python 3.10+ installed → Check by running:

```bash
  python --version
```

Git installed → Check by running:

```bash
  git --version
```

## Setting Up the project

1. **Clone the repository**
   Run this command in whatever directory you want the project

```bash
  git clone https://github.com/BlueRogue12/MediMatch.git
```

## Backend

2. **Create and activate a virtual environment**

```bash
python3 -m venv venc
source venvvenv/bin/activate  # For macOS/Linux
venv\Scripts\Activate     # For Windows (PowerShell)
```

3. **Install dependencies**
```bash
pip install pandas numpy
pip install nltk spacy
pip install scikit-learn imbalanced-learn
pip install matplotlib seaborn
pip install torch torchvision torchaudio
```

## Additional Notes
For the dataset, we used Synthea's 1K Sample Synthetic Patient Records from 2021: https://mitre.box.com/shared/static/aw9po06ypfb9hrau4jamtvtz0e5ziucz.zip

Instructor: Dr. Salim Sazzed
Course: CSCI 5090 – Introduction to Data Science
