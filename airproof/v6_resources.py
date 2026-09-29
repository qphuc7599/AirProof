"""Shared finite contact ledger: raw traffic and receipts spend the same lane."""
from dataclasses import dataclass, field


@dataclass
class ContactBudget:
    capacity: int = 1024
    release_reserved: int = 256
    control_reserved: int = 128
    raw_used: int = 0
    release_used: int = 0
    control_used: int = 0
    charges: list = field(default_factory=list)

    def __post_init__(self):
        values=(self.capacity,self.release_reserved,self.control_reserved)
        if any(type(v) is not int or v<0 for v in values) or sum(values[1:])>values[0]:
            raise ValueError('invalid contact partitions')
        for used, limit in ((self.raw_used,self.capacity-self.release_reserved-self.control_reserved),
                            (self.release_used,self.release_reserved),(self.control_used,self.control_reserved)):
            if type(used) is not int or not 0<=used<=limit:
                raise ValueError('invalid initial lane usage')

    def spend(self, kind: str, payload: bytes) -> bool:
        if kind not in ('raw','receipt','release','control') or not isinstance(payload,bytes):
            raise ValueError('known traffic class and actual serialized bytes required')
        lane='raw' if kind in ('raw','receipt') else kind
        limit={'raw':self.capacity-self.release_reserved-self.control_reserved,
               'release':self.release_reserved,'control':self.control_reserved}[lane]
        used=getattr(self,lane+'_used')
        if used+len(payload)>limit:return False
        setattr(self,lane+'_used',used+len(payload))
        self.charges.append((kind,len(payload)))
        return True

    @property
    def used(self):
        return self.raw_used+self.release_used+self.control_used
