from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict

@dataclass
class PricingResult:
    """
    Standardized output for all pricer classes. This is a sort of "data carrier",
    needed to handle different possible outputs depending on the method adopted for pricing. 
    E.G.: 
    - MonteCarloPricer returns a TUPLE with option price for instance, and confidence interval boundaries;
    - BSMPricer returns the fixed value, a FLOAT, since it exists the analytical solution.
    --> PricingResult handles this problem avoiding data type conflicts, by returning a standardized data type.
    """
    #price is a float
    price: float 
    #whatever is not a price (e.g. confidence intervals) is put into a dictionary, 
    #with the string=key (e.g. "confidence interval") and the corresponding data, 
    # which can be Any type: tuple, ...
    metrics: Dict[str, Any] = field(default_factory=dict)


class BasePricer(ABC):
    """
    Abstract base class for all pricing engines
    """

    @abstractmethod
    def price(self, option, model) -> PricingResult:
        """
        Calculates the price of the given option using the provided market model.
        
        Args:
            option: An instance of a child of BaseOption.
            model: An instance of a child of BaseModel.
            
        Returns:
            The computed price of the option.
        """
        pass