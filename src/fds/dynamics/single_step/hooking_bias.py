from abc import ABC, abstractmethod

class PDFShapingHook(ABC):
    @abstractmethod
    def make_Q(self, space, start_state, predicted_field, *,
               H_prev, H_curr, params: dict) -> "Field":
        """Return Q field over the same space (unit field if no shaping)."""
        pass

class IdentityPDFShaping(PDFShapingHook):
    def make_Q(self, space, start_state, predicted_field, *, H_prev, H_curr, params):
        return predicted_field._unit_field_like()  # unit wrt your Composition

class DeltaOwnGlobalExp(PDFShapingHook):
    def __init__(self, beta: float, delta_transform=None, exp_transform=None):
        self.beta = beta
        self.delta_transform = delta_transform   # optional: make H_curr ⊖ H_prev
        self.exp_transform = exp_transform       # maps scalar field x ↦ exp(beta*x)

    def make_Q(self, space, start_state, predicted_field, *, H_prev, H_curr, params):
        # If you have a Composition inverse, use it; else provide a transform for delta.
        if self.delta_transform is not None:
            dH = self.delta_transform.apply_pair(H_curr, H_prev, params={})
        else:
            dH = H_curr  # fallback: treat current as the driver

        if self.exp_transform is not None:
            Q = self.exp_transform.apply(dH, params={"beta": self.beta})
        else:
            Q = dH  # already scaled

        return Q