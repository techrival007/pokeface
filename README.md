# PokeFace

Find your Pokémon match by analyzing facial features. This project uses computer vision and machine learning to identify which Pokémon you look most like based on facial structure, expressions, and visual characteristics.

## Technical Overview

PokeFace combines web scraping, computer vision, and transformer models to create a face-matching system. The pipeline scrapes Pokémon data and images, extracts visual features using CLIP embeddings, and compares them with human facial features to find matches.

## Interesting Techniques

- **CLIP (Contrastive Language-Image Pre-training)** integration using [Transformers](https://huggingface.co/transformers/) for multi-modal feature extraction
- **Face detection** with [OpenCV's Haar Cascades](https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html)
- **Web scraping** with respectful rate limiting using [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) and [requests](https://docs.python-requests.org/)
- **Weighted similarity scoring** algorithm that considers confidence levels and feature importance
- **Streaming downloads** with chunked file processing for efficient image handling

## Technologies and Libraries

- **[OpenAI CLIP](https://github.com/openai/CLIP)** - Vision transformer for understanding images and text together
- **[PyTorch](https://pytorch.org/)** - Deep learning framework powering the CLIP model
- **[OpenCV](https://opencv.org/)** - Computer vision library for face detection and image processing
- **[Transformers](https://huggingface.co/transformers/)** - Hugging Face library providing pre-trained CLIP models
- **[BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/)** - HTML parsing for web scraping
- **[Pillow](https://pillow.readthedocs.io/)** - Image processing and format conversion

## Project Structure

```
pokeface/
├── images/                    # Downloaded Pokémon sprites
├── extract_features.py
├── match_face.py
├── pokemon_data.json
├── pokemon_features.json
├── requirements.txt
└── scrape_pokemon.py
```

**images/** - Contains scraped Pokémon sprite images organized by Pokédex number and name. Images are downloaded directly from PokemonDB during the scraping process.

The project follows a three-stage pipeline: [`scrape_pokemon.py`](./scrape_pokemon.py) gathers data and images, [`extract_features.py`](./extract_features.py) processes Pokémon images to create feature vectors, and [`match_face.py`](./match_face.py) analyzes user photos to find similar Pokémon matches.
