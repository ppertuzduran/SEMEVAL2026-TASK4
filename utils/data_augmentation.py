"""
Data augmentation module for narrative similarity training.

Implements offline augmentation strategies:
1. T5-based paraphrasing (high quality semantic preservation)
2. Contextual synonym replacement (WordNet-based)

These techniques are proven to improve model generalization on small datasets
by 3-7% according to SimCSE and SBERT literature.
"""

import random
import re
from typing import List, Optional, Tuple
import torch
import nltk
from nltk.corpus import wordnet
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm


# Download required NLTK data
try:
    nltk.data.find('corpora/wordnet.zip')
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger.zip')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)


class ParaphraseAugmenter:
    """
    T5-based paraphrasing augmenter.
    
    Uses a T5 model fine-tuned for paraphrasing to generate semantically
    equivalent variations of input text.
    """
    
    def __init__(
        self,
        model_name: str = "Vamsi/T5_Paraphrase_Paws",
        device: str = None,
        num_beams: int = 5,
        temperature: float = 1.2,
        max_length: int = 256
    ):
        """
        Initialize paraphrase augmenter.
        
        Args:
            model_name: HuggingFace model name (default: Vamsi/T5_Paraphrase_Paws)
            device: Device to run on (auto-detect if None)
            num_beams: Number of beams for beam search
            temperature: Sampling temperature (higher = more diverse)
            max_length: Maximum sequence length
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading paraphrase model: {model_name}")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        self.num_beams = num_beams
        self.temperature = temperature
        self.max_length = max_length
        
        print(f"✓ Paraphrase model loaded on {self.device}")
    
    def augment(self, text: str, n_augmentations: int = 1) -> List[str]:
        """
        Generate paraphrased versions of text.
        
        Args:
            text: Input text
            n_augmentations: Number of paraphrases to generate
            
        Returns:
            List of paraphrased texts
        """
        # Prepare input
        input_text = f"paraphrase: {text}"
        encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # Generate paraphrases
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask'],
                max_length=self.max_length,
                num_beams=self.num_beams,
                num_return_sequences=min(n_augmentations, self.num_beams),
                temperature=self.temperature,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                early_stopping=True
            )
        
        # Decode outputs
        paraphrases = []
        for output in outputs:
            paraphrase = self.tokenizer.decode(output, skip_special_tokens=True)
            # Clean up
            paraphrase = paraphrase.strip()
            if paraphrase and paraphrase != text:
                paraphrases.append(paraphrase)
        
        # If we didn't get enough unique paraphrases, pad with original
        while len(paraphrases) < n_augmentations:
            paraphrases.append(text)
        
        return paraphrases[:n_augmentations]


class SynonymAugmenter:
    """
    Contextual synonym replacement using WordNet.
    
    Replaces words with synonyms while preserving grammatical structure
    and semantic meaning.
    """
    
    def __init__(
        self,
        replacement_prob: float = 0.15,
        max_replacements: int = 3,
        preserve_proper_nouns: bool = True
    ):
        """
        Initialize synonym augmenter.
        
        Args:
            replacement_prob: Probability of replacing each word
            max_replacements: Maximum number of words to replace
            preserve_proper_nouns: Don't replace proper nouns
        """
        self.replacement_prob = replacement_prob
        self.max_replacements = max_replacements
        self.preserve_proper_nouns = preserve_proper_nouns
    
    def _get_synonyms(self, word: str, pos: str) -> List[str]:
        """Get synonyms for a word with specific POS tag."""
        # Map NLTK POS tags to WordNet POS tags
        pos_map = {
            'N': wordnet.NOUN,
            'V': wordnet.VERB,
            'J': wordnet.ADJ,
            'R': wordnet.ADV
        }
        
        wordnet_pos = pos_map.get(pos[0], None)
        if wordnet_pos is None:
            return []
        
        synonyms = set()
        for synset in wordnet.synsets(word, pos=wordnet_pos):
            for lemma in synset.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower():
                    synonyms.add(synonym)
        
        return list(synonyms)
    
    def augment(self, text: str, n_augmentations: int = 1) -> List[str]:
        """
        Generate synonym-replaced versions of text.
        
        Args:
            text: Input text
            n_augmentations: Number of augmented versions to generate
            
        Returns:
            List of augmented texts
        """
        augmented_texts = []
        
        # Tokenize and POS tag
        words = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(words)
        
        for _ in range(n_augmentations):
            new_words = words.copy()
            replacements_made = 0
            
            # Randomly select words to replace
            indices = list(range(len(words)))
            random.shuffle(indices)
            
            for idx in indices:
                if replacements_made >= self.max_replacements:
                    break
                
                word, pos = pos_tags[idx]
                
                # Skip if proper noun and we're preserving them
                if self.preserve_proper_nouns and pos.startswith('NNP'):
                    continue
                
                # Skip punctuation and short words
                if len(word) <= 2 or not word.isalpha():
                    continue
                
                # Replace with probability
                if random.random() < self.replacement_prob:
                    synonyms = self._get_synonyms(word, pos)
                    if synonyms:
                        # Preserve case
                        synonym = random.choice(synonyms)
                        if word[0].isupper():
                            synonym = synonym.capitalize()
                        new_words[idx] = synonym
                        replacements_made += 1
            
            # Reconstruct text
            augmented_text = ' '.join(new_words)
            # Fix spacing around punctuation
            augmented_text = re.sub(r'\s+([.,!?;:])', r'\1', augmented_text)
            augmented_text = re.sub(r'([.,!?;:])\s*', r'\1 ', augmented_text).strip()
            
            augmented_texts.append(augmented_text)
        
        return augmented_texts


class AugmentationPipeline:
    """
    Orchestrates multiple augmentation strategies.
    
    Applies augmenters in sequence or randomly selects one per augmentation.
    """
    
    def __init__(
        self,
        paraphrase_augmenter: Optional[ParaphraseAugmenter] = None,
        synonym_augmenter: Optional[SynonymAugmenter] = None,
        strategy: str = 'random'  # 'random' or 'all'
    ):
        """
        Initialize augmentation pipeline.
        
        Args:
            paraphrase_augmenter: Paraphrase augmenter instance
            synonym_augmenter: Synonym augmenter instance
            strategy: 'random' (pick one augmenter per call) or 'all' (apply all)
        """
        self.augmenters = []
        
        if paraphrase_augmenter:
            self.augmenters.append(('paraphrase', paraphrase_augmenter))
        if synonym_augmenter:
            self.augmenters.append(('synonym', synonym_augmenter))
        
        if not self.augmenters:
            raise ValueError("At least one augmenter must be provided")
        
        self.strategy = strategy
    
    def augment(self, text: str, n_augmentations: int = 1) -> List[str]:
        """
        Generate augmented versions of text.
        
        Args:
            text: Input text
            n_augmentations: Number of augmented versions to generate
            
        Returns:
            List of augmented texts
        """
        if self.strategy == 'random':
            # Randomly select augmenter for each augmentation
            augmented_texts = []
            for _ in range(n_augmentations):
                augmenter_name, augmenter = random.choice(self.augmenters)
                augmented = augmenter.augment(text, n_augmentations=1)
                augmented_texts.extend(augmented)
            return augmented_texts
        
        elif self.strategy == 'all':
            # Apply all augmenters
            augmented_texts = []
            for augmenter_name, augmenter in self.augmenters:
                augmented = augmenter.augment(text, n_augmentations=n_augmentations)
                augmented_texts.extend(augmented)
            return augmented_texts[:n_augmentations]
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def augment_batch(
        self,
        texts: List[str],
        n_augmentations: int = 1,
        show_progress: bool = True
    ) -> List[List[str]]:
        """
        Augment a batch of texts.
        
        Args:
            texts: List of input texts
            n_augmentations: Number of augmentations per text
            show_progress: Show progress bar
            
        Returns:
            List of lists, where each inner list contains augmented versions
        """
        results = []
        iterator = tqdm(texts, desc="Augmenting") if show_progress else texts
        
        for text in iterator:
            augmented = self.augment(text, n_augmentations)
            results.append(augmented)
        
        return results


def create_default_pipeline(device: str = None) -> AugmentationPipeline:
    """
    Create default augmentation pipeline with recommended settings.
    
    Args:
        device: Device to run models on (auto-detect if None)
        
    Returns:
        Configured AugmentationPipeline
    """
    print("Creating augmentation pipeline...")
    
    # Initialize paraphrase augmenter
    paraphrase_aug = ParaphraseAugmenter(
        model_name="Vamsi/T5_Paraphrase_Paws",
        device=device,
        num_beams=5,
        temperature=1.2
    )
    
    # Initialize synonym augmenter
    synonym_aug = SynonymAugmenter(
        replacement_prob=0.15,
        max_replacements=3,
        preserve_proper_nouns=True
    )
    
    # Create pipeline
    pipeline = AugmentationPipeline(
        paraphrase_augmenter=paraphrase_aug,
        synonym_augmenter=synonym_aug,
        strategy='random'  # Randomly pick one augmenter per augmentation
    )
    
    print("✓ Augmentation pipeline ready")
    return pipeline
