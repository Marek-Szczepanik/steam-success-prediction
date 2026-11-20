# Predikce úspěchu her na Steamu

Tento projekt predikuje popularitu hry na **Steamu** z jejích obrázků (header a screenshoty) a metadat pomocí **hlubokého učení**.  
Popularita je vyjádřena jako `y = log(1 + počet pozitivních recenzí + počet negativních recenzí)`. Projekt vznikl jako semestrální práce z předmětu XDEEP.

## Cíle projektu
- Připravit data ze Steamu (metadata a obrázky) a rozdělit je podle roku vydání na trénovací, validační a testovací množinu.
- Zjistit, zda obrázky hry zlepšují predikci popularity oproti samotným metadatům.
- Implementovat a porovnat modely kombinující obrázky a metadata:
  - Model A – vlastní CNN trénovaná od náhodné inicializace
  - Model B – předtrénovaný zmrazený encoder ConvNeXt-Tiny
- Porovnat modely s klasickými přístupy nad metadaty (lineární regrese, gradient boosting).

## Použití projektu
- Všechny skripty se spouštějí z kořene repozitáře.
- Hyperparametry trénování a prostor pro random search jsou v souboru: **config.json**
- Analýza dat a zdůvodnění předzpracování: **eda.ipynb**
- Stažení metadat, čištění a rozdělení dat:
  ```bash
  python -m src.preprocessing.prepare_data
  ```

- Stažení obrázků (stahování lze přerušit a znovu spustit):
  ```bash
  python -m src.preprocessing.download_images
  ```

- Předzpracování obrázků na 128 x 128:
  ```bash
  python -m src.preprocessing.prepare_images
  ```

- Spuštění trénování modelu:
  ```bash
  python src/train.py --model <model>
  # dostupné hodnoty: metadata (výchozí, baseline + MLP), a, b, all (model A a potom model B)
  ```

- Výsledky (průměr a směrodatná odchylka přes seedy 123, 420, 666) se ukládají do složky **results/**, nejlepší checkpointy a průběh trénování jednotlivých běhů do složky **runs/**.

## Použité technologie
- Python 3.12
- torch 2.9 + torchvision 0.24 – trénování neuronových sítí, předtrénovaný ConvNeXt-Tiny
- numpy 2.5 – numerické výpočty
- pandas 3.1 – práce s tabulkovými daty
- scikit-learn 1.9 – baseline modely
- huggingface_hub 2.4 – stažení datasetu
- pillow 13.0 – zpracování obrázků
- matplotlib 3.11 – grafy v EDA

## Hardware
- AMD Radeon RX 9070 XT (PyTorch s ROCm 7.2)
- 32 GB RAM
- Windows

## Data
Projekt využívá dataset v licenci **CC BY 4.0**:
- **Steam Games Dataset** od **Fronkon Games** ([Hugging Face](https://huggingface.co/datasets/FronkonGames/steam-games-dataset))

Obrázky her se stahují ze Steamu a nejsou součástí repozitáře.

## Instalace
Projekt doporučuji spustit v izolovaném virtuálním prostředí:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# nejdříve PyTorch pro dostupnou GPU podle https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```
