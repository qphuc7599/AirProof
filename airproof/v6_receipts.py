"""Signed evidence fragmentation and causal finite direct return queues.

This engine must receive the SAME ContactBudget used by other traffic. It does
not assume gateway contact availability, remote consensus or free control bytes.
"""
from dataclasses import dataclass
import hashlib
import struct
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from .v6_resources import ContactBudget

HEADER=struct.Struct('!4s32sHHI')
FRAME_SIZE=512
CHUNK=FRAME_SIZE-HEADER.size-64


def fragment(payload: bytes, key: Ed25519PrivateKey) -> tuple[bytes,...]:
    if not isinstance(payload,bytes) or not payload or len(payload)>CHUNK*65535:
        raise ValueError('nonempty bounded bytes required')
    identity=hashlib.sha256(payload).digest()
    count=(len(payload)+CHUNK-1)//CHUNK
    result=[]
    for index in range(count):
        part=payload[index*CHUNK:(index+1)*CHUNK]
        body=HEADER.pack(b'AP6R',identity,index,count,len(payload))+part.ljust(CHUNK,b'\0')
        result.append(body+key.sign(body))
    return tuple(result)


class Receiver:
    def __init__(self,key,buffer_bytes=18432):
        if type(buffer_bytes) is not int or buffer_bytes<0:raise ValueError('invalid receiver buffer')
        self.key=key;self.pending={};self.completed={}
        self.buffer_bytes=buffer_bytes;self.peak_buffer=0

    @property
    def occupancy(self):
        # Completed receipts are durable audit storage, accounted separately.
        return sum(len(parts)*FRAME_SIZE for _,parts in self.pending.values())

    def can_accept(self,frame):
        if len(frame)!=FRAME_SIZE:return False
        _,identity,index,_,_=HEADER.unpack(frame[:HEADER.size])
        return (identity in self.completed or index in self.pending.get(identity,(None,{}))[1]
                or self.occupancy+FRAME_SIZE<=self.buffer_bytes)

    def accept(self,frame: bytes):
        if len(frame)!=FRAME_SIZE:raise ValueError('wrong fragment size')
        self.key.verify(frame[-64:],frame[:-64])
        magic,identity,index,count,length=HEADER.unpack(frame[:HEADER.size])
        if magic!=b'AP6R' or not 0<=index<count or count!=(length+CHUNK-1)//CHUNK or length<1:
            raise ValueError('invalid authenticated fragment metadata')
        if identity in self.completed:return identity.hex(),self.completed[identity]
        if not self.can_accept(frame):raise BufferError('receiver reassembly buffer full')
        meta,parts=self.pending.setdefault(identity,((count,length),{}))
        if meta!=(count,length):raise ValueError('inconsistent fragment metadata')
        content=frame[HEADER.size:-64]
        if index in parts and parts[index]!=content:raise ValueError('conflicting fragment')
        parts[index]=content
        self.peak_buffer=max(self.peak_buffer,self.occupancy)
        if len(parts)!=count:return None
        payload=b''.join(parts[i] for i in range(count))[:length]
        if hashlib.sha256(payload).digest()!=identity:raise ValueError('content hash mismatch')
        self.completed[identity]=payload;del self.pending[identity]
        return identity.hex(),payload


@dataclass
class PendingReceipt:
    identity: str
    origin: int
    issued: int
    deadline: int
    frames: tuple[bytes,...]
    next_frame: int = 0


class ReceiptReturnQueue:
    def __init__(self,key: Ed25519PrivateKey,buffer_bytes: int=18432):
        if type(buffer_bytes) is not int or buffer_bytes<0:raise ValueError('invalid buffer')
        self.key=key;self.buffer_bytes=buffer_bytes;self.pending=[];self.seen=set()
        self.delivered={};self.expired=[];self.dropped=[];self.transmitted_bytes=0
        self.control_bytes=0;self.peak_buffer=0;self.last_contact=-1
        self.reservations={};self.peak_committed_buffer=0

    @property
    def occupancy(self):
        return sum((len(r.frames)-r.next_frame)*FRAME_SIZE for r in self.pending)

    @property
    def reserved_bytes(self):
        return sum(self.reservations.values())

    @property
    def committed_occupancy(self):
        return self.occupancy+self.reserved_bytes

    def reserve(self,token,byte_count):
        """Reserve physical queue capacity before promising a future object."""
        if not isinstance(token,str) or not token or token in self.reservations:
            raise ValueError('unique reservation token required')
        if type(byte_count) is not int or byte_count<0:
            raise ValueError('nonnegative integral reservation required')
        if self.committed_occupancy+byte_count>self.buffer_bytes:return False
        self.reservations[token]=byte_count
        self.peak_committed_buffer=max(self.peak_committed_buffer,self.committed_occupancy)
        return True

    def extend_reservation(self,token,additional_bytes):
        """Atomically grow an existing promise reservation."""
        if token not in self.reservations:raise KeyError(token)
        if type(additional_bytes) is not int or additional_bytes<0:
            raise ValueError('nonnegative integral reservation extension required')
        if self.committed_occupancy+additional_bytes>self.buffer_bytes:return False
        self.reservations[token]+=additional_bytes
        self.peak_committed_buffer=max(self.peak_committed_buffer,self.committed_occupancy)
        return True

    def release_reservation(self,token):
        if token not in self.reservations:raise KeyError(token)
        return self.reservations.pop(token)

    def issue(self,payload,origin,epoch,deadline,*,reservation=None):
        if any(type(x) is not int or x<0 for x in (origin,epoch,deadline)) or deadline<epoch:
            raise ValueError('invalid receipt identity clocks')
        frames=fragment(payload,self.key);identity=hashlib.sha256(payload).hexdigest()
        if identity in self.seen:return False
        self.seen.add(identity)
        needed=len(frames)*FRAME_SIZE
        if reservation is None:
            if self.committed_occupancy+needed>self.buffer_bytes:
                self.dropped.append(identity);return False
        else:
            if reservation not in self.reservations:raise KeyError(reservation)
            if needed>self.reservations[reservation]:
                raise BufferError('authenticated object exceeds pre-admission reservation')
            self.reservations[reservation]-=needed
        self.pending.append(PendingReceipt(identity,origin,epoch,deadline,frames))
        self.peak_buffer=max(self.peak_buffer,self.occupancy)
        self.peak_committed_buffer=max(self.peak_committed_buffer,self.committed_occupancy)
        return True

    def expire(self,epoch):
        """Advance deadline accounting even in epochs with no return contact."""
        keep=[]
        for r in self.pending:
            if epoch>r.deadline:self.expired.append(r.identity)
            else:keep.append(r)
        self.pending=keep

    def contact(self,origin,epoch,budget: ContactBudget,receiver: Receiver):
        if epoch<self.last_contact:raise ValueError('contacts must be causal')
        self.last_contact=epoch
        self.expire(epoch)
        for r in list(self.pending):
            if r.origin!=origin or epoch<=r.issued:continue
            frame=r.frames[r.next_frame]
            # Local authenticated-link framing and acknowledgment, not receipt itself.
            control=struct.pack('!4sIQ',b'RACK',r.next_frame,int(r.identity[:16],16))
            # Both allocations must fit before either is consumed.
            if budget.raw_used+len(frame)>budget.capacity-budget.release_reserved-budget.control_reserved:
                return False
            if budget.control_used+len(control)>budget.control_reserved:return False
            if not receiver.can_accept(frame):
                # Backpressure is itself serialized; do not advance fragment state.
                budget.spend('control',control)
                self.control_bytes+=len(control)
                return False
            assert budget.spend('receipt',frame) and budget.spend('control',control)
            result=receiver.accept(frame)
            r.next_frame+=1;self.transmitted_bytes+=len(frame);self.control_bytes+=len(control)
            if r.next_frame==len(r.frames):
                if result is None or result[0]!=r.identity:raise AssertionError('no complete authenticated receipt')
                self.delivered[r.identity]=epoch;self.pending.remove(r)
            return True
        return False
