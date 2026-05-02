"""Result Merger Module.

This module provides functionality to merge and rank results from
multiple search sources (catalog DB and FAISS vector search).
"""
from typing import List, Dict, Optional
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Single search result with metadata."""
    product_id: int
    score: float
    match_type: str  # 'image', 'text', 'metadata', 'hybrid'
    confidence: float
    source_scores: Optional[Dict[str, float]] = None


class ResultMerger:
    """Merges and ranks results from multiple search sources."""
    
    def __init__(
        self,
        catalog_weight: float = 0.6,
        vector_weight: float = 0.4,
        confidence_threshold: float = 0.5,
        low_confidence_catalog_weight: float = 0.3,
        low_confidence_vector_weight: float = 0.7
    ):
        """Initialize the result merger.
        
        Args:
            catalog_weight: Weight for catalog matches (high confidence)
            vector_weight: Weight for vector matches (high confidence)
            confidence_threshold: Threshold to switch weight strategies
            low_confidence_catalog_weight: Catalog weight when confidence is low
            low_confidence_vector_weight: Vector weight when confidence is low
        """
        self.catalog_weight = catalog_weight
        self.vector_weight = vector_weight
        self.confidence_threshold = confidence_threshold
        self.low_confidence_catalog_weight = low_confidence_catalog_weight
        self.low_confidence_vector_weight = low_confidence_vector_weight
        
        logger.info(
            f"ResultMerger initialized (catalog_weight={catalog_weight}, "
            f"vector_weight={vector_weight}, threshold={confidence_threshold})"
        )
    
    def merge(
        self,
        catalog_results: List[Dict],
<<<<<<< Updated upstream
        image_results: List[tuple],
=======
        vector_results: List[tuple],
        text_results: List[tuple],
>>>>>>> Stashed changes
        detection_confidence: float,
        max_results: int = 20,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        text_results: Optional[List[tuple]] = None
    ) -> List[SearchResult]:
<<<<<<< Updated upstream
        """Merge image, text, and catalog search results.

        Implements  α·image_score + β·text_score + γ·meta  re-ranking.

        Args:
            catalog_results: Results from catalog metadata search
            image_results: Results from FAISS image (CLIP) index
            detection_confidence: Confidence from YOLO/OCR detection
            max_results: Maximum number of results to return
            alpha: Dynamic image weight. Overrides internal logic if set.
            beta: Dynamic text weight. Overrides internal logic if set.
            gamma: Dynamic catalog/meta weight. Overrides internal logic if set.
            text_results: Results from FAISS text (BGE-M3) index

        Returns:
            Merged and ranked list of SearchResult objects
        """
        text_results = text_results or []

        # Use dynamic weights from caller if provided (diagram spec)
        # alpha = image_score weight
        # beta  = text_score weight
        # gamma = meta (catalog) weight
        if alpha is not None and beta is not None:
            image_weight = alpha
            text_weight = beta
            catalog_weight = gamma if gamma is not None else max(0.0, 1.0 - alpha - beta)
            logger.debug(
                f"Dynamic weights applied: alpha(image)={alpha}, "
                f"beta(text)={beta}, gamma(meta)={catalog_weight}"
=======
        """Merge catalog, image vector, and text vector search results.
        
        Args:
            catalog_results: Results from catalog metadata search
            vector_results: Results from FAISS image vector search
            text_results: Results from FAISS text vector search
            detection_confidence: Confidence from YOLO/OCR detection
            max_results: Maximum number of results to return
            alpha: Dynamic image/vector weight
            beta: Dynamic text/vector weight
            gamma: Dynamic catalog/metadata weight
            
        Returns:
            Merged and ranked list of SearchResult objects
        """
        # Use dynamic weights from caller if provided
        # alpha = image signal (CLIP vector)
        # beta  = text signal (BGE-M3 vector)
        # gamma = metadata signal (catalog API)
        if alpha is not None and beta is not None and gamma is not None:
            image_weight = alpha
            text_weight = beta
            catalog_weight = gamma
            logger.debug(
                f"Dynamic weights applied: alpha(image)={alpha}, "
                f"beta(text)={beta}, gamma(catalog)={gamma}"
>>>>>>> Stashed changes
            )
        elif detection_confidence < self.confidence_threshold:
            # Low confidence: rely more on visual similarity
            image_weight = self.low_confidence_vector_weight
            text_weight = 0.2
            catalog_weight = self.low_confidence_catalog_weight
<<<<<<< Updated upstream
            image_weight = self.low_confidence_vector_weight
            text_weight = 0.0
            logger.debug(
                f"Low confidence ({detection_confidence:.2f}), "
                f"using weights: catalog={catalog_weight}, image={image_weight}"
=======
            logger.debug(
                f"Low confidence ({detection_confidence:.2f}), "
                f"using weights: image={image_weight}, text={text_weight}, catalog={catalog_weight}"
>>>>>>> Stashed changes
            )
        else:
            image_weight = self.vector_weight
            text_weight = 0.2
            catalog_weight = self.catalog_weight
<<<<<<< Updated upstream
            image_weight = self.vector_weight
            text_weight = 0.0

        # Aggregate scores by product ID
        product_scores = defaultdict(lambda: {"image": 0.0, "text": 0.0, "catalog": 0.0})

=======
        
        # Aggregate scores by product ID
        product_scores = defaultdict(lambda: {"catalog": 0.0, "image": 0.0, "text": 0.0})
        
>>>>>>> Stashed changes
        # Add catalog (metadata) scores
        for result in catalog_results:
            product_id = result.get('product_id')
            if product_id is None:
                continue
            score = result.get('score', 1.0)
            product_scores[product_id]["catalog"] = catalog_weight * score
<<<<<<< Updated upstream

        # Add image (vector) scores
        for product_id, similarity in image_results:
            product_scores[product_id]["image"] = image_weight * similarity

        # Add text scores
        for product_id, similarity in text_results:
            product_scores[product_id]["text"] = text_weight * similarity

        # Calculate combined scores and determine match types
        results = []
        catalog_ids = {r.get('product_id') for r in catalog_results if r.get('product_id')}
        image_ids = {p[0] for p in image_results}
        text_ids = {p[0] for p in text_results}

        for product_id, scores in product_scores.items():
            combined_score = scores["image"] + scores["text"] + scores["catalog"]

            # Determine match type per diagram: image | text | metadata | hybrid
            sources = []
            if product_id in image_ids:
                sources.append('image')
            if product_id in text_ids:
                sources.append('text')
            if product_id in catalog_ids:
                sources.append('metadata')

            if len(sources) >= 2:
                match_type = 'hybrid'
            elif sources:
                match_type = sources[0]
            else:
                match_type = 'unknown'

=======
        
        # Add image vector scores
        for product_id, similarity in vector_results:
            product_scores[product_id]["image"] = image_weight * similarity
        
        # Add text vector scores
        for product_id, similarity in text_results:
            product_scores[product_id]["text"] = text_weight * similarity
        
        # Calculate combined scores and determine match types
        results = []
        catalog_ids = {r.get('product_id') for r in catalog_results if r.get('product_id')}
        image_ids = {p[0] for p in vector_results}
        text_ids = {p[0] for p in text_results}
        
        for product_id, scores in product_scores.items():
            combined_score = scores["catalog"] + scores["image"] + scores["text"]
            
            # Determine match type per flowchart: image | text | metadata | hybrid
            in_catalog = product_id in catalog_ids
            in_image = product_id in image_ids
            in_text = product_id in text_ids
            
            source_count = sum([in_catalog, in_image, in_text])
            
            if source_count >= 2:
                match_type = 'hybrid'
            elif in_catalog:
                match_type = 'metadata'
            elif in_image:
                match_type = 'image'
            else:
                match_type = 'text'
            
>>>>>>> Stashed changes
            results.append(SearchResult(
                product_id=product_id,
                score=combined_score,
                match_type=match_type,
                confidence=detection_confidence,
                source_scores={
<<<<<<< Updated upstream
                    "image": scores["image"],
                    "text": scores["text"],
                    "catalog": scores["catalog"]
=======
                    "catalog": round(scores["catalog"], 3),
                    "image": round(scores["image"], 3),
                    "text": round(scores["text"], 3)
>>>>>>> Stashed changes
                }
            ))

        # Sort by combined score
        results.sort(key=lambda x: x.score, reverse=True)

        # Log top results
        if results:
            top_result = results[0]
            logger.info(
                f"Top result: product_id={top_result.product_id}, "
                f"score={top_result.score:.3f}, match_type={top_result.match_type}"
            )

        return results[:max_results]
    
    def merge_with_diversity(
        self,
        catalog_results: List[Dict],
<<<<<<< Updated upstream
        image_results: List[tuple],
=======
        vector_results: List[tuple],
        text_results: List[tuple],
>>>>>>> Stashed changes
        detection_confidence: float,
        max_results: int = 20,
        diversity_threshold: float = 0.8,
        text_results: Optional[List[tuple]] = None
    ) -> List[SearchResult]:
        """Merge results with diversity to avoid similar products.
        
        Args:
            catalog_results: Results from catalog search
<<<<<<< Updated upstream
            image_results: Results from FAISS image index
=======
            vector_results: Results from FAISS image vector search
            text_results: Results from FAISS text vector search
>>>>>>> Stashed changes
            detection_confidence: Confidence from YOLO/OCR detection
            max_results: Maximum number of results to return
            diversity_threshold: Minimum score difference to include
            text_results: Results from FAISS text index
            
        Returns:
            Diversified list of SearchResult objects
        """
        # Get initial merged results
        all_results = self.merge(
            catalog_results,
<<<<<<< Updated upstream
            image_results,
=======
            vector_results,
            text_results,
>>>>>>> Stashed changes
            detection_confidence,
            max_results=max_results * 2,  # Get more to allow filtering
            text_results=text_results
        )
        
        if not all_results:
            return []
        
        # Apply diversity filtering
        diversified = [all_results[0]]  # Always include top result
        last_score = all_results[0].score
        
        for result in all_results[1:]:
            # Include if score is significantly different or different match type
            if (last_score - result.score) > (1 - diversity_threshold) * last_score:
                diversified.append(result)
                last_score = result.score
            
            if len(diversified) >= max_results:
                break
        
        return diversified
    
    def rerank_by_match_type(
        self,
        results: List[SearchResult],
        type_weights: Optional[Dict[str, float]] = None
    ) -> List[SearchResult]:
        """Re-rank results based on match type preferences.
        
        Args:
            results: List of search results
            type_weights: Weights for each match type
            
        Returns:
            Re-ranked list of results
        """
        if type_weights is None:
            type_weights = {
                'combined': 1.2,  # Boost combined matches
                'catalog': 1.0,
                'vector': 0.9
            }
        
        # Apply type weights
        for result in results:
            weight = type_weights.get(result.match_type, 1.0)
            result.score = result.score * weight
        
        # Re-sort
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results


def format_results_for_response(
    results: List[SearchResult],
    products_data: Optional[Dict[int, Dict]] = None
) -> List[Dict]:
    """Format search results for API response.
    
    Args:
        results: List of SearchResult objects
        products_data: Optional product details by ID
        
    Returns:
        List of formatted result dictionaries
    """
    formatted = []
    
    for result in results:
        item = {
            "product_id": result.product_id,
            "score": round(result.score, 3),
            "match_type": result.match_type,
            "confidence": round(result.confidence, 3)
        }
        
        # Add product details if available
        if products_data and result.product_id in products_data:
            item["product"] = products_data[result.product_id]
        
        formatted.append(item)
    
    return formatted
