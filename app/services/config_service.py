import os
import json
import redis
import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime
from app.models.fakemail import SystemConfig, ProcessingRule

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigService:
    """
    Service for managing system configuration and processing rules
    """
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.system_config_key = "email_processor:system_config"
        self.processing_rules_prefix = "email_processor:rules:"
        self.startup_time = time.time()
        
        # Initialize with default config if not exists
        if not self.redis_client.exists(self.system_config_key):
            self._set_default_config()
    
    def _set_default_config(self) -> None:
        """
        Set default system configuration in Redis
        """
        config = SystemConfig(
            worker_count=int(os.getenv("PROCESSOR_WORKER_COUNT", "3")),
            max_retries=int(os.getenv("PROCESSOR_MAX_RETRIES", "3")),
            retry_delay_ms=int(os.getenv("PROCESSOR_RETRY_DELAY_MS", "1000")),
            max_concurrent_requests=int(os.getenv("MAX_CONCURRENT_REQUESTS", "10")),
            request_timeout_seconds=int(os.getenv("REQUEST_TIMEOUT_SECONDS", "10")),
            batch_size=int(os.getenv("BATCH_SIZE", "20")),
            worker_idle_timeout_seconds=int(os.getenv("WORKER_IDLE_TIMEOUT_SECONDS", "300")),
            throttle_threshold=int(os.getenv("THROTTLE_THRESHOLD", "100")),
            default_priority=int(os.getenv("DEFAULT_PRIORITY", "3"))
        )
        self.redis_client.set(self.system_config_key, config.model_dump_json())
        logger.info("Default system configuration initialized")
        
    def get_system_config(self) -> SystemConfig:
        """
        Get the current system configuration
        """
        config_data = self.redis_client.get(self.system_config_key)
        if not config_data:
            self._set_default_config()
            config_data = self.redis_client.get(self.system_config_key)
            
        config_dict = json.loads(config_data)
        return SystemConfig(**config_dict)
        
    def update_system_config(self, config_update: Dict[str, Any]) -> SystemConfig:
        """
        Update the system configuration
        """
        current_config = self.get_system_config()
        current_dict = current_config.model_dump()
        
        # Update only the provided fields
        for key, value in config_update.items():
            if key in current_dict:
                current_dict[key] = value
                
        updated_config = SystemConfig(**current_dict)
        self.redis_client.set(self.system_config_key, updated_config.model_dump_json())
        
        logger.info(f"System configuration updated: {config_update}")
        return updated_config
        
    def get_uptime_seconds(self) -> int:
        """
        Get system uptime in seconds
        """
        return int(time.time() - self.startup_time)
        
    def add_processing_rule(self, rule: ProcessingRule) -> str:
        """
        Add a new email processing rule
        """
        rule_key = f"{self.processing_rules_prefix}{rule.rule_id}"
        self.redis_client.set(rule_key, rule.model_dump_json())
        
        # Add to rules index
        self.redis_client.sadd("email_processor:rules", rule.rule_id)
        
        logger.info(f"Added processing rule: {rule.name} ({rule.rule_id})")
        return rule.rule_id
        
    def get_processing_rule(self, rule_id: str) -> Optional[ProcessingRule]:
        """
        Get a processing rule by ID
        """
        rule_key = f"{self.processing_rules_prefix}{rule_id}"
        rule_data = self.redis_client.get(rule_key)
        
        if not rule_data:
            return None
            
        rule_dict = json.loads(rule_data)
        return ProcessingRule(**rule_dict)
        
    def update_processing_rule(self, rule_id: str, rule_update: Dict[str, Any]) -> Optional[ProcessingRule]:
        """
        Update a processing rule
        """
        rule = self.get_processing_rule(rule_id)
        if not rule:
            return None
            
        rule_dict = rule.model_dump()
        
        # Update only the provided fields
        for key, value in rule_update.items():
            if key in rule_dict:
                rule_dict[key] = value
                
        updated_rule = ProcessingRule(**rule_dict)
        rule_key = f"{self.processing_rules_prefix}{rule_id}"
        self.redis_client.set(rule_key, updated_rule.model_dump_json())
        
        logger.info(f"Updated processing rule: {updated_rule.name} ({rule_id})")
        return updated_rule
        
    def delete_processing_rule(self, rule_id: str) -> bool:
        """
        Delete a processing rule
        """
        rule_key = f"{self.processing_rules_prefix}{rule_id}"
        if not self.redis_client.exists(rule_key):
            return False
            
        self.redis_client.delete(rule_key)
        self.redis_client.srem("email_processor:rules", rule_id)
        
        logger.info(f"Deleted processing rule: {rule_id}")
        return True
        
    def get_all_processing_rules(self) -> Dict[str, ProcessingRule]:
        """
        Get all processing rules
        """
        rule_ids = self.redis_client.smembers("email_processor:rules")
        rules = {}
        
        for rule_id in rule_ids:
            rule_id_str = rule_id.decode("utf-8")
            rule = self.get_processing_rule(rule_id_str)
            if rule:
                rules[rule_id_str] = rule
                
        return rules
        
    def get_active_processing_rules(self) -> Dict[str, ProcessingRule]:
        """
        Get all active processing rules
        """
        all_rules = self.get_all_processing_rules()
        return {rule_id: rule for rule_id, rule in all_rules.items() if rule.active}