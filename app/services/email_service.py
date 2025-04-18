import json
import redis
from typing import List, Optional, Dict, Any
from app.models.email import Email, EmailFilter
from datetime import datetime
import time

class EmailService:
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.email_key_prefix = "email:"
        self.email_list_key = "emails"
    
    def create_email(self, email: Email) -> str:
        """Store an email in the system and return its ID"""
        email_dict = email.model_dump(mode='json')
        
        # Format datetime for JSON serialization
        email_dict["received_at"] = email_dict["received_at"].isoformat()
        
        # Store the email
        email_key = f"{self.email_key_prefix}{email.id}"
        self.redis_client.set(email_key, json.dumps(email_dict))
        
        # Add to the email list with timestamp for sorting
        self.redis_client.zadd(self.email_list_key, {email.id: time.time()})
        
        # Add to indexes for filtering
        for recipient in email.recipients:
            self.redis_client.sadd(f"recipient:{recipient}", email.id)
        
        self.redis_client.sadd(f"sender:{email.sender}", email.id)
        
        # Index labels
        for label in email.labels:
            self.redis_client.sadd(f"label:{label}", email.id)
        
        # Publish event for real-time notifications
        self.redis_client.publish("email:new", json.dumps({
            "id": email.id,
            "sender": email.sender,
            "subject": email.subject,
            "recipients": email.recipients,
            "timestamp": datetime.now().isoformat()
        }))
            
        return email.id
    
    def get_email(self, email_id: str) -> Optional[Email]:
        """Retrieve an email by ID"""
        email_key = f"{self.email_key_prefix}{email_id}"
        email_data = self.redis_client.get(email_key)
        
        if not email_data:
            return None
        
        email_dict = json.loads(email_data)
        
        # Parse datetime from string
        email_dict["received_at"] = datetime.fromisoformat(email_dict["received_at"])
        
        return Email(**email_dict)
    
    def list_emails(self, 
                  filter_params: Optional[EmailFilter] = None, 
                  offset: int = 0, 
                  limit: int = 50) -> List[Email]:
        """List emails with filtering and pagination"""
        # Get all email IDs sorted by received time (newest first)
        email_ids = self.redis_client.zrevrange(self.email_list_key, offset, offset + limit - 1)
        
        # Apply filters if provided
        filtered_ids = email_ids
        if filter_params:
            if filter_params.sender:
                sender_ids = self.redis_client.smembers(f"sender:{filter_params.sender}")
                filtered_ids = [id for id in filtered_ids if id in sender_ids]
                
            if filter_params.recipient:
                recipient_ids = self.redis_client.smembers(f"recipient:{filter_params.recipient}")
                filtered_ids = [id for id in filtered_ids if id in recipient_ids]
                
            if filter_params.label:
                label_ids = self.redis_client.smembers(f"label:{filter_params.label}")
                filtered_ids = [id for id in filtered_ids if id in label_ids]
        
        # Retrieve emails
        emails = []
        for email_id in filtered_ids:
            email = self.get_email(email_id.decode('utf-8'))
            if email:
                # Apply subject filter
                if filter_params and filter_params.subject and filter_params.subject.lower() not in email.subject.lower():
                    continue
                    
                # Apply date filters
                if filter_params and filter_params.date_from and email.received_at < filter_params.date_from:
                    continue
                    
                if filter_params and filter_params.date_to and email.received_at > filter_params.date_to:
                    continue
                    
                # Apply attachment filter
                if (filter_params and filter_params.has_attachments is not None and 
                    bool(email.attachments) != filter_params.has_attachments):
                    continue
                    
                emails.append(email)
                
        return emails
    
    def delete_email(self, email_id: str) -> bool:
        """Delete an email by ID"""
        email = self.get_email(email_id)
        if not email:
            return False
            
        # Remove from indexes
        for recipient in email.recipients:
            self.redis_client.srem(f"recipient:{recipient}", email_id)
            
        self.redis_client.srem(f"sender:{email.sender}", email_id)
        
        for label in email.labels:
            self.redis_client.srem(f"label:{label}", email_id)
            
        # Remove from email list
        self.redis_client.zrem(self.email_list_key, email_id)
        
        # Delete the email itself
        email_key = f"{self.email_key_prefix}{email_id}"
        self.redis_client.delete(email_key)
        
        return True
        
    def mark_as_read(self, email_id: str) -> bool:
        """Mark an email as read"""
        email = self.get_email(email_id)
        if not email:
            return False
            
        email.read = True
        
        # Update the email in Redis
        email_dict = email.model_dump(mode='json')
        email_dict["received_at"] = email_dict["received_at"].isoformat()
        
        email_key = f"{self.email_key_prefix}{email_id}"
        self.redis_client.set(email_key, json.dumps(email_dict))
        
        return True
        
    def add_label(self, email_id: str, label: str) -> bool:
        """Add a label to an email"""
        email = self.get_email(email_id)
        if not email:
            return False
            
        if label not in email.labels:
            email.labels.append(label)
            
            # Update the email in Redis
            email_dict = email.model_dump(mode='json')
            email_dict["received_at"] = email_dict["received_at"].isoformat()
            
            email_key = f"{self.email_key_prefix}{email_id}"
            self.redis_client.set(email_key, json.dumps(email_dict))
            
            # Update the label index
            self.redis_client.sadd(f"label:{label}", email_id)
            
        return True
        
    def remove_label(self, email_id: str, label: str) -> bool:
        """Remove a label from an email"""
        email = self.get_email(email_id)
        if not email:
            return False
            
        if label in email.labels:
            email.labels.remove(label)
            
            # Update the email in Redis
            email_dict = email.model_dump(mode='json')
            email_dict["received_at"] = email_dict["received_at"].isoformat()
            
            email_key = f"{self.email_key_prefix}{email_id}"
            self.redis_client.set(email_key, json.dumps(email_dict))
            
            # Update the label index
            self.redis_client.srem(f"label:{label}", email_id)
            
        return True