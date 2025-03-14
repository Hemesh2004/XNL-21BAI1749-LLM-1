import hashlib
import hmac
import os
from cryptography.fernet import Fernet
from typing import Dict, Any
import json

class SecurityManager:
    def __init__(self, key_path: str = ".env"):
        """Initialize security manager with encryption keys"""
        self.encryption_key = os.getenv("MODEL_ENCRYPTION_KEY") or Fernet.generate_key()
        self.watermark_key = os.getenv("WATERMARK_KEY") or os.urandom(32)
        self.fernet = Fernet(self.encryption_key)
        
    def watermark_model(self, model_state: Dict[str, Any]) -> Dict[str, Any]:
        """Add watermark to model weights"""
        serialized = json.dumps(model_state, sort_keys=True)
        watermark = hmac.new(self.watermark_key, 
                           serialized.encode(),
                           hashlib.sha256).hexdigest()
        
        model_state["_watermark"] = watermark
        return model_state
    
    def verify_watermark(self, model_state: Dict[str, Any]) -> bool:
        """Verify model watermark"""
        if "_watermark" not in model_state:
            return False
            
        original_watermark = model_state.pop("_watermark")
        serialized = json.dumps(model_state, sort_keys=True)
        computed_watermark = hmac.new(self.watermark_key,
                                    serialized.encode(),
                                    hashlib.sha256).hexdigest()
                                    
        model_state["_watermark"] = original_watermark
        return hmac.compare_digest(original_watermark, computed_watermark)
    
    def encrypt_model(self, model_path: str) -> None:
        """Encrypt model file using AES-256"""
        with open(model_path, 'rb') as f:
            data = f.read()
        
        encrypted_data = self.fernet.encrypt(data)
        
        with open(f"{model_path}.encrypted", 'wb') as f:
            f.write(encrypted_data)
            
    def decrypt_model(self, encrypted_path: str) -> None:
        """Decrypt model file"""
        with open(encrypted_path, 'rb') as f:
            encrypted_data = f.read()
            
        decrypted_data = self.fernet.decrypt(encrypted_data)
        
        output_path = encrypted_path.replace('.encrypted', '')
        with open(output_path, 'wb') as f:
            f.write(decrypted_data)
