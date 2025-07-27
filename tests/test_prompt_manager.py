import pytest
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from src.services.langchain.prompt_manager import PromptManager, prompt_manager


class TestPromptManager:
    """Test cases for PromptManager"""

    @pytest.fixture
    def manager(self):
        """Create a new PromptManager instance for testing"""
        return PromptManager()

    def test_init(self, manager):
        """Test PromptManager initialization"""
        # Verify prompts are initialized
        assert len(manager._prompts) > 0
        
        # Verify all expected prompts are present
        expected_prompts = [
            "hyde_generation",
            "summary_refinement", 
            "final_report",
            "rag_query"
        ]
        for prompt_name in expected_prompts:
            assert prompt_name in manager._prompts

    def test_get_prompt_success(self, manager):
        """Test getting an existing prompt"""
        prompt = manager.get_prompt("hyde_generation")
        assert prompt is not None
        assert isinstance(prompt, ChatPromptTemplate)
        
        # Verify prompt can be formatted with expected variables
        formatted = prompt.format(monitoring_data={"test": "data"})
        assert "監控數據" in formatted
        # 修正：比對 Python 物件的字串表示法
        assert "{'test': 'data'}" in formatted

    def test_get_prompt_not_found(self, manager):
        """Test getting a non-existent prompt"""
        with pytest.raises(ValueError, match="Prompt 'non_existent' not found"):
            manager.get_prompt("non_existent")

    def test_add_custom_prompt_chat_template(self, manager):
        """Test adding a custom chat prompt template"""
        template = "This is a custom template with {variable}"
        manager.add_custom_prompt("custom_chat", template)
        
        # Verify prompt was added
        assert "custom_chat" in manager._prompts
        prompt = manager.get_prompt("custom_chat")
        assert isinstance(prompt, ChatPromptTemplate)
        
        # Test formatting
        formatted = prompt.format(variable="test_value")
        assert "This is a custom template with test_value" in formatted

    def test_add_custom_prompt_with_variables(self, manager):
        """Test adding a custom prompt with explicit input variables"""
        template = "Template with {var1} and {var2}"
        manager.add_custom_prompt(
            "custom_vars", 
            template, 
            input_variables=["var1", "var2"]
        )
        
        # Verify prompt was added
        assert "custom_vars" in manager._prompts
        prompt = manager.get_prompt("custom_vars")
        assert isinstance(prompt, PromptTemplate)
        assert prompt.input_variables == ["var1", "var2"]
        
        # Test formatting
        formatted = prompt.format(var1="first", var2="second")
        assert "Template with first and second" == formatted

    def test_list_prompts(self, manager):
        """Test listing all available prompts"""
        prompts = manager.list_prompts()
        
        assert isinstance(prompts, list)
        assert len(prompts) >= 4  # At least the default prompts
        
        # Verify default prompts are included
        expected = ["hyde_generation", "summary_refinement", "final_report", "rag_query"]
        for prompt_name in expected:
            assert prompt_name in prompts

    def test_update_prompt_chat_template(self, manager):
        """Test updating an existing chat prompt template"""
        original_prompt = manager.get_prompt("rag_query")
        
        # Update the prompt
        new_template = "Updated template: {question} - {context}"
        manager.update_prompt("rag_query", new_template)
        
        # Verify prompt was updated
        updated_prompt = manager.get_prompt("rag_query")
        formatted = updated_prompt.format(question="Q", context="C")
        assert "Updated template: Q - C" in formatted

    def test_update_prompt_with_variables(self, manager):
        """Test updating a prompt that has explicit input variables"""
        # First add a prompt with variables
        manager.add_custom_prompt(
            "test_update",
            "Original: {var1}",
            input_variables=["var1", "var2"]
        )
        
        # Update it
        manager.update_prompt("test_update", "Updated: {var1} and {var2}")
        
        # Verify variables are preserved
        prompt = manager.get_prompt("test_update")
        assert prompt.input_variables == ["var1", "var2"]
        formatted = prompt.format(var1="A", var2="B")
        assert formatted == "Updated: A and B"

    def test_update_nonexistent_prompt(self, manager):
        """Test updating a non-existent prompt (should not raise error)"""
        # This should not raise an error, just do nothing
        manager.update_prompt("non_existent", "new template")
        
        # Verify prompt was not added
        assert "non_existent" not in manager._prompts

    def test_prompt_manager_singleton(self):
        """Test that prompt_manager is a singleton instance"""
        assert isinstance(prompt_manager, PromptManager)
        assert len(prompt_manager._prompts) > 0

    def test_all_default_prompts_format_correctly(self, manager):
        """Test that all default prompts can be formatted correctly"""
        test_data = {
            "monitoring_data": {"host": "test-host", "cpu": 80},
            "context": "Test context",
            "question": "Test question"
        }
        
        # Test hyde_generation
        hyde_prompt = manager.get_prompt("hyde_generation")
        hyde_formatted = hyde_prompt.format(monitoring_data=test_data["monitoring_data"])
        assert "test-host" in hyde_formatted
        
        # Test summary_refinement
        summary_prompt = manager.get_prompt("summary_refinement")
        summary_formatted = summary_prompt.format(
            monitoring_data=test_data["monitoring_data"],
            context=test_data["context"]
        )
        assert "Test context" in summary_formatted
        
        # Test final_report
        final_prompt = manager.get_prompt("final_report")
        final_formatted = final_prompt.format(
            monitoring_data=test_data["monitoring_data"],
            context=test_data["context"]
        )
        assert "洞見分析" in final_formatted
        assert "具體建議" in final_formatted
        
        # Test rag_query
        rag_prompt = manager.get_prompt("rag_query")
        rag_formatted = rag_prompt.format(
            question=test_data["question"],
            context=test_data["context"]
        )
        assert "Test question" in rag_formatted