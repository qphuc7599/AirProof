// SPDX-License-Identifier: MIT
pragma solidity 0.8.28;

/// @title AirProofAnchor
/// @notice Stores batch commitments and policy history, never citizen identifiers or locations.
contract AirProofAnchor {
    uint256 public constant MAX_BATCH_COUNT = 1_000_000;
    uint256 public constant MAX_FUTURE_EPOCHS = 24;

    address public immutable owner;
    mapping(address => bool) public batchers;
    mapping(address => uint256) public revokedEffectiveEpoch;
    mapping(bytes32 => bytes32) public policySchema;
    mapping(bytes32 => bool) public rootExists;

    struct Anchor {
        bytes32 root;
        uint64 count;
        bytes32 schemaHash;
        bytes32 policyHash;
        bytes32 cidDigest;
        address submitter;
        uint64 anchoredAt;
    }

    mapping(bytes32 => Anchor) private anchors;
    mapping(bytes32 => bytes32) public correctionPredecessor;

    error Unauthorized();
    error InvalidValue();
    error UnknownPolicy();
    error DuplicateAnchor();
    error RevokedBatcher();
    error UnknownRoot();

    event BatcherAuthorization(address indexed batcher, bool authorized, uint256 effectiveEpoch);
    event PolicyRegistered(bytes32 indexed policyHash, bytes32 indexed schemaHash);
    event BatchAnchored(
        bytes32 indexed cityId,
        uint256 indexed epoch,
        bytes32 indexed root,
        uint256 count,
        bytes32 schemaHash,
        bytes32 policyHash,
        bytes32 cidDigest
    );
    event BatchCorrected(bytes32 indexed previousRoot, bytes32 indexed newRoot, bytes32 reasonHash);

    constructor() {
        owner = msg.sender;
        batchers[msg.sender] = true;
        emit BatcherAuthorization(msg.sender, true, 0);
    }

    modifier onlyOwner() {
        if (msg.sender != owner) revert Unauthorized();
        _;
    }

    modifier activeBatcher(uint256 epoch) {
        if (!batchers[msg.sender]) revert Unauthorized();
        uint256 revokedAt = revokedEffectiveEpoch[msg.sender];
        if (revokedAt != 0 && epoch >= revokedAt) revert RevokedBatcher();
        _;
    }

    function setBatcher(address batcher, bool authorized) external onlyOwner {
        if (batcher == address(0)) revert InvalidValue();
        batchers[batcher] = authorized;
        if (authorized) revokedEffectiveEpoch[batcher] = 0;
        emit BatcherAuthorization(batcher, authorized, 0);
    }

    function registerPolicy(bytes32 policyHash, bytes32 schemaHash) external onlyOwner {
        if (policyHash == bytes32(0) || schemaHash == bytes32(0)) revert InvalidValue();
        if (policySchema[policyHash] != bytes32(0)) revert DuplicateAnchor();
        policySchema[policyHash] = schemaHash;
        emit PolicyRegistered(policyHash, schemaHash);
    }

    function anchorBatch(
        bytes32 cityId,
        uint256 epoch,
        bytes32 root,
        uint256 count,
        bytes32 schemaHash,
        bytes32 policyHash,
        bytes32 cidDigest
    ) external activeBatcher(epoch) {
        if (cityId == bytes32(0) || root == bytes32(0) || cidDigest == bytes32(0)) revert InvalidValue();
        if (count == 0 || count > MAX_BATCH_COUNT || epoch > block.timestamp / 1 hours + MAX_FUTURE_EPOCHS) {
            revert InvalidValue();
        }
        if (policySchema[policyHash] == bytes32(0) || policySchema[policyHash] != schemaHash) {
            revert UnknownPolicy();
        }
        bytes32 key = anchorKey(cityId, epoch);
        if (anchors[key].root != bytes32(0) || rootExists[root]) revert DuplicateAnchor();
        anchors[key] = Anchor(root, uint64(count), schemaHash, policyHash, cidDigest, msg.sender, uint64(block.timestamp));
        rootExists[root] = true;
        emit BatchAnchored(cityId, epoch, root, count, schemaHash, policyHash, cidDigest);
    }

    function anchorCorrection(bytes32 previousRoot, bytes32 newRoot, bytes32 reasonHash, uint256 epoch)
        external
        activeBatcher(epoch)
    {
        if (!rootExists[previousRoot]) revert UnknownRoot();
        if (newRoot == bytes32(0) || reasonHash == bytes32(0) || rootExists[newRoot]) revert InvalidValue();
        correctionPredecessor[newRoot] = previousRoot;
        rootExists[newRoot] = true;
        emit BatchCorrected(previousRoot, newRoot, reasonHash);
    }

    function revokeBatcher(address batcher, uint256 effectiveEpoch) external onlyOwner {
        if (!batchers[batcher] || effectiveEpoch == 0) revert InvalidValue();
        revokedEffectiveEpoch[batcher] = effectiveEpoch;
        emit BatcherAuthorization(batcher, false, effectiveEpoch);
    }

    function getAnchor(bytes32 cityId, uint256 epoch) external view returns (Anchor memory) {
        return anchors[anchorKey(cityId, epoch)];
    }

    function anchorKey(bytes32 cityId, uint256 epoch) public pure returns (bytes32) {
        return keccak256(abi.encode(cityId, epoch));
    }
}
