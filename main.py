from langflow.base.data.utils import IMG_FILE_TYPES, TEXT_FILE_TYPES
from langflow.base.io.chat import ChatComponent
from langflow.inputs import BoolInput
from langflow.io import DropdownInput, FileInput, MessageTextInput, MultilineInput, Output
from langflow.schema.message import Message
from langflow.utils.constants import MESSAGE_SENDER_AI, MESSAGE_SENDER_NAME_USER, MESSAGE_SENDER_USER


class ChatInput(ChatComponent):
    display_name = "Chat Input"
    description = "Get chat inputs from the Playground."
    icon = "MessagesSquare"
    name = "ChatInput"

    inputs = [
        MultilineInput(
            name="input_value",
            display_name="Text",
            value="",
            info="Message to be passed as input.",
        ),
        BoolInput(
            name="should_store_message",
            display_name="Store Messages",
            info="Store the message in the history.",
            value=True,
            advanced=True,
        ),
        DropdownInput(
            name="sender",
            display_name="Sender Type",
            options=[MESSAGE_SENDER_AI, MESSAGE_SENDER_USER],
            value=MESSAGE_SENDER_USER,
            info="Type of sender.",
            advanced=True,
        ),
        MessageTextInput(
            name="sender_name",
            display_name="Sender Name",
            info="Name of the sender.",
            value=MESSAGE_SENDER_NAME_USER,
            advanced=True,
        ),
        MessageTextInput(
            name="session_id",
            display_name="Session ID",
            info="The session ID of the chat. If empty, the current session ID parameter will be used.",
            advanced=True,
        ),
        FileInput(
            name="files",
            display_name="Files",
            file_types=TEXT_FILE_TYPES + IMG_FILE_TYPES,
            info="Files to be sent with the message.",
            advanced=True,
            is_list=True,
        ),
        MessageTextInput(
            name="background_color",
            display_name="Background Color",
            info="The background color of the icon.",
            advanced=True,
        ),
        MessageTextInput(
            name="chat_icon",
            display_name="Icon",
            info="The icon of the message.",
            advanced=True,
        ),
        MessageTextInput(
            name="text_color",
            display_name="Text Color",
            info="The text color of the name",
            advanced=True,
        ),
    ]
    outputs = [
        Output(display_name="Message", name="message", method="message_response"),
    ]

    def message_response(self) -> Message:
        _background_color = self.background_color
        _text_color = self.text_color
        _icon = self.chat_icon
        message = Message(
            text=self.input_value,
            sender=self.sender,
            sender_name=self.sender_name,
            session_id=self.session_id,
            files=self.files,
            properties={"background_color": _background_color, "text_color": _text_color, "icon": _icon},
        )
        if self.session_id and isinstance(message, Message) and self.should_store_message:
            stored_message = self.send_message(
                message,
            )
            self.message.value = stored_message
            message = stored_message

        self.status = message
        return message
from typing import Any

from langflow.base.embeddings.model import LCEmbeddingsModel
from langflow.field_typing import Embeddings
from langflow.inputs.inputs import DropdownInput, SecretStrInput
from langflow.io import FloatInput, MessageTextInput
from langflow.schema.dotdict import dotdict


class NVIDIAEmbeddingsComponent(LCEmbeddingsModel):
    display_name: str = "NVIDIA Embeddings"
    description: str = "Generate embeddings using NVIDIA models."
    icon = "NVIDIA"

    inputs = [
        DropdownInput(
            name="model",
            display_name="Model",
            options=[
                "nvidia/nv-embed-v1",
                "snowflake/arctic-embed-I",
            ],
            value="nvidia/nv-embed-v1",
        ),
        MessageTextInput(
            name="base_url",
            display_name="NVIDIA Base URL",
            refresh_button=True,
            value="https://integrate.api.nvidia.com/v1",
        ),
        SecretStrInput(
            name="nvidia_api_key",
            display_name="NVIDIA API Key",
            info="The NVIDIA API Key.",
            advanced=False,
            value="NVIDIA_API_KEY",
        ),
        FloatInput(
            name="temperature",
            display_name="Model Temperature",
            value=0.1,
            advanced=True,
        ),
    ]

    def update_build_config(self, build_config: dotdict, field_value: Any, field_name: str | None = None):
        if field_name == "base_url" and field_value:
            try:
                build_model = self.build_embeddings()
                ids = [model.id for model in build_model.available_models]
                build_config["model"]["options"] = ids
                build_config["model"]["value"] = ids[0]
            except Exception as e:
                msg = f"Error getting model names: {e}"
                raise ValueError(msg) from e
        return build_config

    def build_embeddings(self) -> Embeddings:
        try:
            from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
        except ImportError as e:
            msg = "Please install langchain-nvidia-ai-endpoints to use the Nvidia model."
            raise ImportError(msg) from e
        try:
            output = NVIDIAEmbeddings(
                model=self.model,
                base_url=self.base_url,
                temperature=self.temperature,
                nvidia_api_key=self.nvidia_api_key,
            )
        except Exception as e:
            msg = f"Could not connect to NVIDIA API. Error: {e}"
            raise ValueError(msg) from e
        return output
import os
from collections import defaultdict

import orjson
from astrapy import DataAPIClient
from astrapy.admin import parse_api_endpoint
from langchain_astradb import AstraDBVectorStore

from langflow.base.vectorstores.model import LCVectorStoreComponent, check_cached_vector_store
from langflow.helpers import docs_to_data
from langflow.inputs import DictInput, FloatInput, MessageTextInput, NestedDictInput
from langflow.io import (
    BoolInput,
    DataInput,
    DropdownInput,
    HandleInput,
    IntInput,
    MultilineInput,
    SecretStrInput,
    StrInput,
)
from langflow.schema import Data


class AstraVectorStoreComponent(LCVectorStoreComponent):
    display_name: str = "Astra DB"
    description: str = "Implementation of Vector Store using Astra DB with search capabilities"
    documentation: str = "https://docs.langflow.org/starter-projects-vector-store-rag"
    name = "AstraDB"
    icon: str = "AstraDB"

    _cached_vector_store: AstraDBVectorStore | None = None

    VECTORIZE_PROVIDERS_MAPPING = defaultdict(
        list,
        {
            "Azure OpenAI": [
                "azureOpenAI",
                ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
            ],
            "Hugging Face - Dedicated": ["huggingfaceDedicated", ["endpoint-defined-model"]],
            "Hugging Face - Serverless": [
                "huggingface",
                [
                    "sentence-transformers/all-MiniLM-L6-v2",
                    "intfloat/multilingual-e5-large",
                    "intfloat/multilingual-e5-large-instruct",
                    "BAAI/bge-small-en-v1.5",
                    "BAAI/bge-base-en-v1.5",
                    "BAAI/bge-large-en-v1.5",
                ],
            ],
            "Jina AI": [
                "jinaAI",
                [
                    "jina-embeddings-v2-base-en",
                    "jina-embeddings-v2-base-de",
                    "jina-embeddings-v2-base-es",
                    "jina-embeddings-v2-base-code",
                    "jina-embeddings-v2-base-zh",
                ],
            ],
            "Mistral AI": ["mistral", ["mistral-embed"]],
            "NVIDIA": ["nvidia", ["NV-Embed-QA"]],
            "OpenAI": ["openai", ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]],
            "Upstage": ["upstageAI", ["solar-embedding-1-large"]],
            "Voyage AI": [
                "voyageAI",
                ["voyage-large-2-instruct", "voyage-law-2", "voyage-code-2", "voyage-large-2", "voyage-2"],
            ],
        },
    )

    inputs = [
        SecretStrInput(
            name="token",
            display_name="Astra DB Application Token",
            info="Authentication token for accessing Astra DB.",
            value="ASTRA_DB_APPLICATION_TOKEN",
            required=True,
            advanced=os.getenv("ASTRA_ENHANCED", "false").lower() == "true",
        ),
        SecretStrInput(
            name="api_endpoint",
            display_name="Database" if os.getenv("ASTRA_ENHANCED", "false").lower() == "true" else "API Endpoint",
            info="API endpoint URL for the Astra DB service.",
            value="ASTRA_DB_API_ENDPOINT",
            required=True,
        ),
        StrInput(
            name="collection_name",
            display_name="Collection Name",
            info="The name of the collection within Astra DB where the vectors will be stored.",
            required=True,
        ),
        MultilineInput(
            name="search_input",
            display_name="Search Input",
        ),
        DataInput(
            name="ingest_data",
            display_name="Ingest Data",
            is_list=True,
        ),
        StrInput(
            name="keyspace",
            display_name="Keyspace",
            info="Optional keyspace within Astra DB to use for the collection.",
            advanced=True,
        ),
        DropdownInput(
            name="embedding_choice",
            display_name="Embedding Model or Astra Vectorize",
            info="Determines whether to use Astra Vectorize for the collection.",
            options=["Embedding Model", "Astra Vectorize"],
            real_time_refresh=True,
            value="Embedding Model",
        ),
        HandleInput(
            name="embedding_model",
            display_name="Embedding Model",
            input_types=["Embeddings"],
            info="Allows an embedding model configuration.",
        ),
        DropdownInput(
            name="metric",
            display_name="Metric",
            info="Optional distance metric for vector comparisons in the vector store.",
            options=["cosine", "dot_product", "euclidean"],
            value="cosine",
            advanced=True,
        ),
        IntInput(
            name="batch_size",
            display_name="Batch Size",
            info="Optional number of data to process in a single batch.",
            advanced=True,
        ),
        IntInput(
            name="bulk_insert_batch_concurrency",
            display_name="Bulk Insert Batch Concurrency",
            info="Optional concurrency level for bulk insert operations.",
            advanced=True,
        ),
        IntInput(
            name="bulk_insert_overwrite_concurrency",
            display_name="Bulk Insert Overwrite Concurrency",
            info="Optional concurrency level for bulk insert operations that overwrite existing data.",
            advanced=True,
        ),
        IntInput(
            name="bulk_delete_concurrency",
            display_name="Bulk Delete Concurrency",
            info="Optional concurrency level for bulk delete operations.",
            advanced=True,
        ),
        DropdownInput(
            name="setup_mode",
            display_name="Setup Mode",
            info="Configuration mode for setting up the vector store, with options like 'Sync' or 'Off'.",
            options=["Sync", "Off"],
            advanced=True,
            value="Sync",
        ),
        BoolInput(
            name="pre_delete_collection",
            display_name="Pre Delete Collection",
            info="Boolean flag to determine whether to delete the collection before creating a new one.",
            advanced=True,
        ),
        StrInput(
            name="metadata_indexing_include",
            display_name="Metadata Indexing Include",
            info="Optional list of metadata fields to include in the indexing.",
            is_list=True,
            advanced=True,
        ),
        StrInput(
            name="metadata_indexing_exclude",
            display_name="Metadata Indexing Exclude",
            info="Optional list of metadata fields to exclude from the indexing.",
            is_list=True,
            advanced=True,
        ),
        StrInput(
            name="collection_indexing_policy",
            display_name="Collection Indexing Policy",
            info='Optional JSON string for the "indexing" field of the collection. '
            "See https://docs.datastax.com/en/astra-db-serverless/api-reference/collections.html#the-indexing-option",
            advanced=True,
        ),
        IntInput(
            name="number_of_results",
            display_name="Number of Results",
            info="Number of results to return.",
            advanced=True,
            value=4,
        ),
        DropdownInput(
            name="search_type",
            display_name="Search Type",
            info="Search type to use",
            options=["Similarity", "Similarity with score threshold", "MMR (Max Marginal Relevance)"],
            value="Similarity",
            advanced=True,
        ),
        FloatInput(
            name="search_score_threshold",
            display_name="Search Score Threshold",
            info="Minimum similarity score threshold for search results. "
            "(when using 'Similarity with score threshold')",
            value=0,
            advanced=True,
        ),
        NestedDictInput(
            name="advanced_search_filter",
            display_name="Search Metadata Filter",
            info="Optional dictionary of filters to apply to the search query.",
            advanced=True,
        ),
        DictInput(
            name="search_filter",
            display_name="[DEPRECATED] Search Metadata Filter",
            info="Deprecated: use advanced_search_filter. Optional dictionary of filters to apply to the search query.",
            advanced=True,
            is_list=True,
        ),
    ]

    def del_fields(self, build_config, field_list):
        for field in field_list:
            if field in build_config:
                del build_config[field]

        return build_config

    def insert_in_dict(self, build_config, field_name, new_parameters):
        # Insert the new key-value pair after the found key
        for new_field_name, new_parameter in new_parameters.items():
            # Get all the items as a list of tuples (key, value)
            items = list(build_config.items())

            # Find the index of the key to insert after
            idx = len(items)
            for i, (key, _) in enumerate(items):
                if key == field_name:
                    idx = i + 1
                    break

            items.insert(idx, (new_field_name, new_parameter))

            # Clear the original dictionary and update with the modified items
            build_config.clear()
            build_config.update(items)

        return build_config

    def update_providers_mapping(self):
        # If we don't have token or api_endpoint, we can't fetch the list of providers
        if not self.token or not self.api_endpoint:
            self.log("Astra DB token and API endpoint are required to fetch the list of Vectorize providers.")

            return self.VECTORIZE_PROVIDERS_MAPPING

        try:
            self.log("Dynamically updating list of Vectorize providers.")

            # Get the admin object
            client = DataAPIClient(token=self.token)
            admin = client.get_admin()

            # Get the embedding providers
            db_admin = admin.get_database_admin(self.api_endpoint)
            embedding_providers = db_admin.find_embedding_providers().as_dict()

            vectorize_providers_mapping = {}

            # Map the provider display name to the provider key and models
            for provider_key, provider_data in embedding_providers["embeddingProviders"].items():
                display_name = provider_data["displayName"]
                models = [model["name"] for model in provider_data["models"]]

                vectorize_providers_mapping[display_name] = [provider_key, models]

            # Sort the resulting dictionary
            return defaultdict(list, dict(sorted(vectorize_providers_mapping.items())))
        except Exception as e:  # noqa: BLE001
            self.log(f"Error fetching Vectorize providers: {e}")

            return self.VECTORIZE_PROVIDERS_MAPPING

    def update_build_config(self, build_config: dict, field_value: str, field_name: str | None = None):
        if field_name == "embedding_choice":
            if field_value == "Astra Vectorize":
                self.del_fields(build_config, ["embedding_model"])

                # Update the providers mapping
                vectorize_providers = self.update_providers_mapping()

                new_parameter = DropdownInput(
                    name="embedding_provider",
                    display_name="Embedding Provider",
                    options=vectorize_providers.keys(),
                    value="",
                    required=True,
                    real_time_refresh=True,
                ).to_dict()

                self.insert_in_dict(build_config, "embedding_choice", {"embedding_provider": new_parameter})
            else:
                self.del_fields(
                    build_config,
                    [
                        "embedding_provider",
                        "model",
                        "z_01_model_parameters",
                        "z_02_api_key_name",
                        "z_03_provider_api_key",
                        "z_04_authentication",
                    ],
                )

                new_parameter = HandleInput(
                    name="embedding_model",
                    display_name="Embedding Model",
                    input_types=["Embeddings"],
                    info="Allows an embedding model configuration.",
                ).to_dict()

                self.insert_in_dict(build_config, "embedding_choice", {"embedding_model": new_parameter})

        elif field_name == "embedding_provider":
            self.del_fields(
                build_config,
                ["model", "z_01_model_parameters", "z_02_api_key_name", "z_03_provider_api_key", "z_04_authentication"],
            )

            # Update the providers mapping
            vectorize_providers = self.update_providers_mapping()
            model_options = vectorize_providers[field_value][1]

            new_parameter = DropdownInput(
                name="model",
                display_name="Model",
                info="The embedding model to use for the selected provider. Each provider has a different set of "
                "models available (full list at "
                "https://docs.datastax.com/en/astra-db-serverless/databases/embedding-generation.html):\n\n"
                f"{', '.join(model_options)}",
                options=model_options,
                value=None,
                required=True,
                real_time_refresh=True,
            ).to_dict()

            self.insert_in_dict(build_config, "embedding_provider", {"model": new_parameter})

        elif field_name == "model":
            self.del_fields(
                build_config,
                ["z_01_model_parameters", "z_02_api_key_name", "z_03_provider_api_key", "z_04_authentication"],
            )

            new_parameter_1 = DictInput(
                name="z_01_model_parameters",
                display_name="Model Parameters",
                is_list=True,
            ).to_dict()

            new_parameter_2 = MessageTextInput(
                name="z_02_api_key_name",
                display_name="API Key Name",
                info="The name of the embeddings provider API key stored on Astra. "
                "If set, it will override the 'ProviderKey' in the authentication parameters.",
            ).to_dict()

            new_parameter_3 = SecretStrInput(
                load_from_db=False,
                name="z_03_provider_api_key",
                display_name="Provider API Key",
                info="An alternative to the Astra Authentication that passes an API key for the provider "
                "with each request to Astra DB. "
                "This may be used when Vectorize is configured for the collection, "
                "but no corresponding provider secret is stored within Astra's key management system.",
            ).to_dict()

            new_parameter_4 = DictInput(
                name="z_04_authentication",
                display_name="Authentication Parameters",
                is_list=True,
            ).to_dict()

            self.insert_in_dict(
                build_config,
                "model",
                {
                    "z_01_model_parameters": new_parameter_1,
                    "z_02_api_key_name": new_parameter_2,
                    "z_03_provider_api_key": new_parameter_3,
                    "z_04_authentication": new_parameter_4,
                },
            )

        return build_config

    def build_vectorize_options(self, **kwargs):
        for attribute in [
            "embedding_provider",
            "model",
            "z_01_model_parameters",
            "z_02_api_key_name",
            "z_03_provider_api_key",
            "z_04_authentication",
        ]:
            if not hasattr(self, attribute):
                setattr(self, attribute, None)

        # Fetch values from kwargs if any self.* attributes are None
        provider_value = self.VECTORIZE_PROVIDERS_MAPPING.get(self.embedding_provider, [None])[0] or kwargs.get(
            "embedding_provider"
        )
        model_name = self.model or kwargs.get("model")
        authentication = {**(self.z_04_authentication or kwargs.get("z_04_authentication", {}))}
        parameters = self.z_01_model_parameters or kwargs.get("z_01_model_parameters", {})

        # Set the API key name if provided
        api_key_name = self.z_02_api_key_name or kwargs.get("z_02_api_key_name")
        provider_key = self.z_03_provider_api_key or kwargs.get("z_03_provider_api_key")
        if api_key_name:
            authentication["providerKey"] = api_key_name

        # Set authentication and parameters to None if no values are provided
        if not authentication:
            authentication = None
        if not parameters:
            parameters = None

        return {
            # must match astrapy.info.CollectionVectorServiceOptions
            "collection_vector_service_options": {
                "provider": provider_value,
                "modelName": model_name,
                "authentication": authentication,
                "parameters": parameters,
            },
            "collection_embedding_api_key": provider_key,
        }

    @check_cached_vector_store
    def build_vector_store(self, vectorize_options=None):
        try:
            from langchain_astradb import AstraDBVectorStore
            from langchain_astradb.utils.astradb import SetupMode
        except ImportError as e:
            msg = (
                "Could not import langchain Astra DB integration package. "
                "Please install it with `pip install langchain-astradb`."
            )
            raise ImportError(msg) from e

        try:
            if not self.setup_mode:
                self.setup_mode = self._inputs["setup_mode"].options[0]

            setup_mode_value = SetupMode[self.setup_mode.upper()]
        except KeyError as e:
            msg = f"Invalid setup mode: {self.setup_mode}"
            raise ValueError(msg) from e

        if self.embedding_choice == "Embedding Model":
            embedding_dict = {"embedding": self.embedding_model}
        else:
            from astrapy.info import CollectionVectorServiceOptions

            # Fetch values from kwargs if any self.* attributes are None
            dict_options = vectorize_options or self.build_vectorize_options()

            # Set the embedding dictionary
            embedding_dict = {
                "collection_vector_service_options": CollectionVectorServiceOptions.from_dict(
                    dict_options.get("collection_vector_service_options")
                ),
                "collection_embedding_api_key": dict_options.get("collection_embedding_api_key"),
            }

        try:
            vector_store = AstraDBVectorStore(
                collection_name=self.collection_name,
                token=self.token,
                api_endpoint=self.api_endpoint,
                namespace=self.keyspace or None,
                environment=parse_api_endpoint(self.api_endpoint).environment if self.api_endpoint else None,
                metric=self.metric or None,
                batch_size=self.batch_size or None,
                bulk_insert_batch_concurrency=self.bulk_insert_batch_concurrency or None,
                bulk_insert_overwrite_concurrency=self.bulk_insert_overwrite_concurrency or None,
                bulk_delete_concurrency=self.bulk_delete_concurrency or None,
                setup_mode=setup_mode_value,
                pre_delete_collection=self.pre_delete_collection,
                metadata_indexing_include=[s for s in self.metadata_indexing_include if s] or None,
                metadata_indexing_exclude=[s for s in self.metadata_indexing_exclude if s] or None,
                collection_indexing_policy=orjson.dumps(self.collection_indexing_policy)
                if self.collection_indexing_policy
                else None,
                **embedding_dict,
            )
        except Exception as e:
            msg = f"Error initializing AstraDBVectorStore: {e}"
            raise ValueError(msg) from e

        self._add_documents_to_vector_store(vector_store)

        return vector_store

    def _add_documents_to_vector_store(self, vector_store) -> None:
        documents = []
        for _input in self.ingest_data or []:
            if isinstance(_input, Data):
                documents.append(_input.to_lc_document())
            else:
                msg = "Vector Store Inputs must be Data objects."
                raise TypeError(msg)

        if documents:
            self.log(f"Adding {len(documents)} documents to the Vector Store.")
            try:
                vector_store.add_documents(documents)
            except Exception as e:
                msg = f"Error adding documents to AstraDBVectorStore: {e}"
                raise ValueError(msg) from e
        else:
            self.log("No documents to add to the Vector Store.")

    def _map_search_type(self) -> str:
        if self.search_type == "Similarity with score threshold":
            return "similarity_score_threshold"
        if self.search_type == "MMR (Max Marginal Relevance)":
            return "mmr"
        return "similarity"

    def _build_search_args(self):
        query = self.search_input if isinstance(self.search_input, str) and self.search_input.strip() else None
        search_filter = (
            {k: v for k, v in self.search_filter.items() if k and v and k.strip()} if self.search_filter else None
        )

        if query:
            args = {
                "query": query,
                "search_type": self._map_search_type(),
                "k": self.number_of_results,
                "score_threshold": self.search_score_threshold,
            }
        elif self.advanced_search_filter or search_filter:
            args = {
                "n": self.number_of_results,
            }
        else:
            return {}

        filter_arg = self.advanced_search_filter or {}

        if search_filter:
            self.log(self.log(f"`search_filter` is deprecated. Use `advanced_search_filter`. Cleaned: {search_filter}"))
            filter_arg.update(search_filter)

        if filter_arg:
            args["filter"] = filter_arg

        return args

    def search_documents(self, vector_store=None) -> list[Data]:
        vector_store = vector_store or self.build_vector_store()

        self.log(f"Search input: {self.search_input}")
        self.log(f"Search type: {self.search_type}")
        self.log(f"Number of results: {self.number_of_results}")

        try:
            search_args = self._build_search_args()
        except Exception as e:
            msg = f"Error in AstraDBVectorStore._build_search_args: {e}"
            raise ValueError(msg) from e

        if not search_args:
            self.log("No search input or filters provided. Skipping search.")
            return []

        docs = []
        search_method = "search" if "query" in search_args else "metadata_search"

        try:
            self.log(f"Calling vector_store.{search_method} with args: {search_args}")
            docs = getattr(vector_store, search_method)(**search_args)
        except Exception as e:
            msg = f"Error performing {search_method} in AstraDBVectorStore: {e}"
            raise ValueError(msg) from e

        self.log(f"Retrieved documents: {len(docs)}")

        data = docs_to_data(docs)
        self.log(f"Converted documents to data: {len(data)}")
        self.status = data
        return data

    def get_retriever_kwargs(self):
        search_args = self._build_search_args()
        return {
            "search_type": self._map_search_type(),
            "search_kwargs": search_args,
        }
from langchain.memory import ConversationBufferMemory

from langflow.custom import Component
from langflow.field_typing import BaseChatMemory
from langflow.helpers.data import data_to_text
from langflow.inputs import HandleInput
from langflow.io import DropdownInput, IntInput, MessageTextInput, MultilineInput, Output
from langflow.memory import LCBuiltinChatMemory, get_messages
from langflow.schema import Data
from langflow.schema.message import Message
from langflow.utils.constants import MESSAGE_SENDER_AI, MESSAGE_SENDER_USER


class MemoryComponent(Component):
    display_name = "Message History"
    description = "Retrieves stored chat messages from Langflow tables or an external memory."
    icon = "message-square-more"
    name = "Memory"

    inputs = [
        HandleInput(
            name="memory",
            display_name="External Memory",
            input_types=["BaseChatMessageHistory"],
            info="Retrieve messages from an external memory. If empty, it will use the Langflow tables.",
        ),
        DropdownInput(
            name="sender",
            display_name="Sender Type",
            options=[MESSAGE_SENDER_AI, MESSAGE_SENDER_USER, "Machine and User"],
            value="Machine and User",
            info="Filter by sender type.",
            advanced=True,
        ),
        MessageTextInput(
            name="sender_name",
            display_name="Sender Name",
            info="Filter by sender name.",
            advanced=True,
        ),
        IntInput(
            name="n_messages",
            display_name="Number of Messages",
            value=100,
            info="Number of messages to retrieve.",
            advanced=True,
        ),
        MessageTextInput(
            name="session_id",
            display_name="Session ID",
            info="The session ID of the chat. If empty, the current session ID parameter will be used.",
            advanced=True,
        ),
        DropdownInput(
            name="order",
            display_name="Order",
            options=["Ascending", "Descending"],
            value="Ascending",
            info="Order of the messages.",
            advanced=True,
        ),
        MultilineInput(
            name="template",
            display_name="Template",
            info="The template to use for formatting the data. "
            "It can contain the keys {text}, {sender} or any other key in the message data.",
            value="{sender_name}: {text}",
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Data", name="messages", method="retrieve_messages"),
        Output(display_name="Text", name="messages_text", method="retrieve_messages_as_text"),
    ]

    def retrieve_messages(self) -> Data:
        sender = self.sender
        sender_name = self.sender_name
        session_id = self.session_id
        n_messages = self.n_messages
        order = "DESC" if self.order == "Descending" else "ASC"

        if sender == "Machine and User":
            sender = None

        if self.memory:
            # override session_id
            self.memory.session_id = session_id

            stored = self.memory.messages
            # langchain memories are supposed to return messages in ascending order
            if order == "DESC":
                stored = stored[::-1]
            if n_messages:
                stored = stored[:n_messages]
            stored = [Message.from_lc_message(m) for m in stored]
            if sender:
                expected_type = MESSAGE_SENDER_AI if sender == MESSAGE_SENDER_AI else MESSAGE_SENDER_USER
                stored = [m for m in stored if m.type == expected_type]
        else:
            stored = get_messages(
                sender=sender,
                sender_name=sender_name,
                session_id=session_id,
                limit=n_messages,
                order=order,
            )
        self.status = stored
        return stored

    def retrieve_messages_as_text(self) -> Message:
        stored_text = data_to_text(self.template, self.retrieve_messages())
        self.status = stored_text
        return Message(text=stored_text)

    def build_lc_memory(self) -> BaseChatMemory:
        chat_memory = self.memory or LCBuiltinChatMemory(flow_id=self.flow_id, session_id=self.session_id)
        return ConversationBufferMemory(chat_memory=chat_memory)
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZipFile, is_zipfile

from langflow.base.data.utils import TEXT_FILE_TYPES, parallel_load_data, parse_text_file_to_data
from langflow.custom import Component
from langflow.io import BoolInput, FileInput, IntInput, Output
from langflow.schema import Data


class FileComponent(Component):
    """Handles loading of individual or zipped text files.

    Processes multiple valid files within a zip archive if provided.

    Attributes:
        display_name: Display name of the component.
        description: Brief component description.
        icon: Icon to represent the component.
        name: Identifier for the component.
        inputs: Inputs required by the component.
        outputs: Output of the component after processing files.
    """

    display_name = "File"
    description = "Load a file to be used in your project."
    icon = "file-text"
    name = "File"

    inputs = [
        FileInput(
            name="path",
            display_name="Path",
            file_types=[*TEXT_FILE_TYPES, "zip"],
            info=f"Supported file types: {', '.join([*TEXT_FILE_TYPES, 'zip'])}",
        ),
        BoolInput(
            name="silent_errors",
            display_name="Silent Errors",
            advanced=True,
            info="If true, errors will not raise an exception.",
        ),
        BoolInput(
            name="use_multithreading",
            display_name="Use Multithreading",
            advanced=True,
            info="If true, parallel processing will be enabled for zip files.",
        ),
        IntInput(
            name="concurrency_multithreading",
            display_name="Multithreading Concurrency",
            advanced=True,
            info="The maximum number of workers to use, if concurrency is enabled",
            value=4,
        ),
    ]

    outputs = [Output(display_name="Data", name="data", method="load_file")]

    def load_file(self) -> Data:
        """Load and parse file(s) from a zip archive.

        Raises:
            ValueError: If no file is uploaded or file path is invalid.

        Returns:
            Data: Parsed data from file(s).
        """
        # Check if the file path is provided
        if not self.path:
            self.log("File path is missing.")
            msg = "Please upload a file for processing."

            raise ValueError(msg)

        resolved_path = Path(self.resolve_path(self.path))
        try:
            # Check if the file is a zip archive
            if is_zipfile(resolved_path):
                self.log(f"Processing zip file: {resolved_path.name}.")

                return self._process_zip_file(
                    resolved_path,
                    silent_errors=self.silent_errors,
                    parallel=self.use_multithreading,
                )

            self.log(f"Processing single file: {resolved_path.name}.")

            return self._process_single_file(resolved_path, silent_errors=self.silent_errors)
        except FileNotFoundError:
            self.log(f"File not found: {resolved_path.name}.")

            raise

    def _process_zip_file(self, zip_path: Path, *, silent_errors: bool = False, parallel: bool = False) -> Data:
        """Process text files within a zip archive.

        Args:
            zip_path: Path to the zip file.
            silent_errors: Suppresses errors if True.
            parallel: Enables parallel processing if True.

        Returns:
            list[Data]: Combined data from all valid files.

        Raises:
            ValueError: If no valid files found in the archive.
        """
        data: list[Data] = []
        with ZipFile(zip_path, "r") as zip_file:
            # Filter file names based on extensions in TEXT_FILE_TYPES and ignore hidden files
            valid_files = [
                name
                for name in zip_file.namelist()
                if (
                    any(name.endswith(ext) for ext in TEXT_FILE_TYPES)
                    and not name.startswith("__MACOSX")
                    and not name.startswith(".")
                )
            ]

            # Raise an error if no valid files found
            if not valid_files:
                self.log("No valid files in the zip archive.")

                # Return empty data if silent_errors is True
                if silent_errors:
                    return data  # type: ignore[return-value]

                # Raise an error if no valid files found
                msg = "No valid files in the zip archive."
                raise ValueError(msg)

            # Define a function to process each file
            def process_file(file_name, silent_errors=silent_errors):
                with NamedTemporaryFile(delete=False) as temp_file:
                    temp_path = Path(temp_file.name).with_name(file_name)
                    with zip_file.open(file_name) as file_content:
                        temp_path.write_bytes(file_content.read())
                try:
                    return self._process_single_file(temp_path, silent_errors=silent_errors)
                finally:
                    temp_path.unlink()

            # Process files in parallel if specified
            if parallel:
                self.log(
                    f"Initializing parallel Thread Pool Executor with max workers: "
                    f"{self.concurrency_multithreading}."
                )

                # Process files in parallel
                initial_data = parallel_load_data(
                    valid_files,
                    silent_errors=silent_errors,
                    load_function=process_file,
                    max_concurrency=self.concurrency_multithreading,
                )

                # Filter out empty data
                data = list(filter(None, initial_data))
            else:
                # Sequential processing
                data = [process_file(file_name) for file_name in valid_files]

        self.log(f"Successfully processed zip file: {zip_path.name}.")

        return data  # type: ignore[return-value]

    def _process_single_file(self, file_path: Path, *, silent_errors: bool = False) -> Data:
        """Process a single file.

        Args:
            file_path: Path to the file.
            silent_errors: Suppresses errors if True.

        Returns:
            Data: Parsed data from the file.

        Raises:
            ValueError: For unsupported file formats.
        """
        # Check if the file type is supported
        if not any(file_path.suffix == ext for ext in ["." + f for f in TEXT_FILE_TYPES]):
            self.log(f"Unsupported file type: {file_path.suffix}")

            # Return empty data if silent_errors is True
            if silent_errors:
                return Data()

            msg = f"Unsupported file type: {file_path.suffix}"
            raise ValueError(msg)

        try:
            # Parse the text file as appropriate
            data = parse_text_file_to_data(str(file_path), silent_errors=silent_errors)  # type: ignore[assignment]
            if not data:
                data = Data()

            self.log(f"Successfully processed file: {file_path.name}.")
        except Exception as e:
            self.log(f"Error processing file {file_path.name}: {e}")

            # Return empty data if silent_errors is True
            if not silent_errors:
                raise

            data = Data()

        return data
from langflow.custom import Component
from langflow.helpers.data import data_to_text
from langflow.io import DataInput, MultilineInput, Output, StrInput
from langflow.schema.message import Message


class ParseDataComponent(Component):
    display_name = "Parse Data"
    description = "Convert Data into plain text following a specified template."
    icon = "braces"
    name = "ParseData"

    inputs = [
        DataInput(name="data", display_name="Data", info="The data to convert to text."),
        MultilineInput(
            name="template",
            display_name="Template",
            info="The template to use for formatting the data. "
            "It can contain the keys {text}, {data} or any other key in the Data.",
            value="{text}",
        ),
        StrInput(name="sep", display_name="Separator", advanced=True, value="\n"),
    ]

    outputs = [
        Output(display_name="Text", name="text", method="parse_data"),
    ]

    def parse_data(self) -> Message:
        data = self.data if isinstance(self.data, list) else [self.data]
        template = self.template

        result_string = data_to_text(template, data, sep=self.sep)
        self.status = result_string
        return Message(text=result_string)
from langchain_text_splitters import CharacterTextSplitter

from langflow.custom import Component
from langflow.io import HandleInput, IntInput, MessageTextInput, Output
from langflow.schema import Data
from langflow.utils.util import unescape_string


class SplitTextComponent(Component):
    display_name: str = "Split Text"
    description: str = "Split text into chunks based on specified criteria."
    icon = "scissors-line-dashed"
    name = "SplitText"

    inputs = [
        HandleInput(
            name="data_inputs",
            display_name="Data Inputs",
            info="The data to split.",
            input_types=["Data"],
            is_list=True,
        ),
        IntInput(
            name="chunk_overlap",
            display_name="Chunk Overlap",
            info="Number of characters to overlap between chunks.",
            value=200,
        ),
        IntInput(
            name="chunk_size",
            display_name="Chunk Size",
            info="The maximum number of characters in each chunk.",
            value=1000,
        ),
        MessageTextInput(
            name="separator",
            display_name="Separator",
            info="The character to split on. Defaults to newline.",
            value="\n",
        ),
    ]

    outputs = [
        Output(display_name="Chunks", name="chunks", method="split_text"),
    ]

    def _docs_to_data(self, docs):
        return [Data(text=doc.page_content, data=doc.metadata) for doc in docs]

    def split_text(self) -> list[Data]:
        separator = unescape_string(self.separator)

        documents = [_input.to_lc_document() for _input in self.data_inputs if isinstance(_input, Data)]

        splitter = CharacterTextSplitter(
            chunk_overlap=self.chunk_overlap,
            chunk_size=self.chunk_size,
            separator=separator,
        )
        docs = splitter.split_documents(documents)
        data = self._docs_to_data(docs)
        self.status = data
        return data
from typing import Any

from langflow.base.embeddings.model import LCEmbeddingsModel
from langflow.field_typing import Embeddings
from langflow.inputs.inputs import DropdownInput, SecretStrInput
from langflow.io import FloatInput, MessageTextInput
from langflow.schema.dotdict import dotdict


class NVIDIAEmbeddingsComponent(LCEmbeddingsModel):
    display_name: str = "NVIDIA Embeddings"
    description: str = "Generate embeddings using NVIDIA models."
    icon = "NVIDIA"

    inputs = [
        DropdownInput(
            name="model",
            display_name="Model",
            options=[
                "nvidia/nv-embed-v1",
                "snowflake/arctic-embed-I",
            ],
            value="nvidia/nv-embed-v1",
        ),
        MessageTextInput(
            name="base_url",
            display_name="NVIDIA Base URL",
            refresh_button=True,
            value="https://integrate.api.nvidia.com/v1",
        ),
        SecretStrInput(
            name="nvidia_api_key",
            display_name="NVIDIA API Key",
            info="The NVIDIA API Key.",
            advanced=False,
            value="NVIDIA_API_KEY",
        ),
        FloatInput(
            name="temperature",
            display_name="Model Temperature",
            value=0.1,
            advanced=True,
        ),
    ]

    def update_build_config(self, build_config: dotdict, field_value: Any, field_name: str | None = None):
        if field_name == "base_url" and field_value:
            try:
                build_model = self.build_embeddings()
                ids = [model.id for model in build_model.available_models]
                build_config["model"]["options"] = ids
                build_config["model"]["value"] = ids[0]
            except Exception as e:
                msg = f"Error getting model names: {e}"
                raise ValueError(msg) from e
        return build_config

    def build_embeddings(self) -> Embeddings:
        try:
            from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
        except ImportError as e:
            msg = "Please install langchain-nvidia-ai-endpoints to use the Nvidia model."
            raise ImportError(msg) from e
        try:
            output = NVIDIAEmbeddings(
                model=self.model,
                base_url=self.base_url,
                temperature=self.temperature,
                nvidia_api_key=self.nvidia_api_key,
            )
        except Exception as e:
            msg = f"Could not connect to NVIDIA API. Error: {e}"
            raise ValueError(msg) from e
        return output
from langflow.base.prompts.api_utils import process_prompt_template
from langflow.custom import Component
from langflow.inputs.inputs import DefaultPromptField
from langflow.io import Output, PromptInput
from langflow.schema.message import Message
from langflow.template.utils import update_template_values


class PromptComponent(Component):
    display_name: str = "Prompt"
    description: str = "Create a prompt template with dynamic variables."
    icon = "prompts"
    trace_type = "prompt"
    name = "Prompt"

    inputs = [
        PromptInput(name="template", display_name="Template"),
    ]

    outputs = [
        Output(display_name="Prompt Message", name="prompt", method="build_prompt"),
    ]

    async def build_prompt(self) -> Message:
        prompt = Message.from_template(**self._attributes)
        self.status = prompt.text
        return prompt

    def _update_template(self, frontend_node: dict):
        prompt_template = frontend_node["template"]["template"]["value"]
        custom_fields = frontend_node["custom_fields"]
        frontend_node_template = frontend_node["template"]
        _ = process_prompt_template(
            template=prompt_template,
            name="template",
            custom_fields=custom_fields,
            frontend_node_template=frontend_node_template,
        )
        return frontend_node

    def post_code_processing(self, new_frontend_node: dict, current_frontend_node: dict):
        """This function is called after the code validation is done."""
        frontend_node = super().post_code_processing(new_frontend_node, current_frontend_node)
        template = frontend_node["template"]["template"]["value"]
        # Kept it duplicated for backwards compatibility
        _ = process_prompt_template(
            template=template,
            name="template",
            custom_fields=frontend_node["custom_fields"],
            frontend_node_template=frontend_node["template"],
        )
        # Now that template is updated, we need to grab any values that were set in the current_frontend_node
        # and update the frontend_node with those values
        update_template_values(new_template=frontend_node, previous_template=current_frontend_node["template"])
        return frontend_node

    def _get_fallback_input(self, **kwargs):
        return DefaultPromptField(**kwargs)
import os
from collections import defaultdict

import orjson
from astrapy import DataAPIClient
from astrapy.admin import parse_api_endpoint
from langchain_astradb import AstraDBVectorStore

from langflow.base.vectorstores.model import LCVectorStoreComponent, check_cached_vector_store
from langflow.helpers import docs_to_data
from langflow.inputs import DictInput, FloatInput, MessageTextInput, NestedDictInput
from langflow.io import (
    BoolInput,
    DataInput,
    DropdownInput,
    HandleInput,
    IntInput,
    MultilineInput,
    SecretStrInput,
    StrInput,
)
from langflow.schema import Data


class AstraVectorStoreComponent(LCVectorStoreComponent):
    display_name: str = "Astra DB"
    description: str = "Implementation of Vector Store using Astra DB with search capabilities"
    documentation: str = "https://docs.langflow.org/starter-projects-vector-store-rag"
    name = "AstraDB"
    icon: str = "AstraDB"

    _cached_vector_store: AstraDBVectorStore | None = None

    VECTORIZE_PROVIDERS_MAPPING = defaultdict(
        list,
        {
            "Azure OpenAI": [
                "azureOpenAI",
                ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
            ],
            "Hugging Face - Dedicated": ["huggingfaceDedicated", ["endpoint-defined-model"]],
            "Hugging Face - Serverless": [
                "huggingface",
                [
                    "sentence-transformers/all-MiniLM-L6-v2",
                    "intfloat/multilingual-e5-large",
                    "intfloat/multilingual-e5-large-instruct",
                    "BAAI/bge-small-en-v1.5",
                    "BAAI/bge-base-en-v1.5",
                    "BAAI/bge-large-en-v1.5",
                ],
            ],
            "Jina AI": [
                "jinaAI",
                [
                    "jina-embeddings-v2-base-en",
                    "jina-embeddings-v2-base-de",
                    "jina-embeddings-v2-base-es",
                    "jina-embeddings-v2-base-code",
                    "jina-embeddings-v2-base-zh",
                ],
            ],
            "Mistral AI": ["mistral", ["mistral-embed"]],
            "NVIDIA": ["nvidia", ["NV-Embed-QA"]],
            "OpenAI": ["openai", ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]],
            "Upstage": ["upstageAI", ["solar-embedding-1-large"]],
            "Voyage AI": [
                "voyageAI",
                ["voyage-large-2-instruct", "voyage-law-2", "voyage-code-2", "voyage-large-2", "voyage-2"],
            ],
        },
    )

    inputs = [
        SecretStrInput(
            name="token",
            display_name="Astra DB Application Token",
            info="Authentication token for accessing Astra DB.",
            value="ASTRA_DB_APPLICATION_TOKEN",
            required=True,
            advanced=os.getenv("ASTRA_ENHANCED", "false").lower() == "true",
        ),
        SecretStrInput(
            name="api_endpoint",
            display_name="Database" if os.getenv("ASTRA_ENHANCED", "false").lower() == "true" else "API Endpoint",
            info="API endpoint URL for the Astra DB service.",
            value="ASTRA_DB_API_ENDPOINT",
            required=True,
        ),
        StrInput(
            name="collection_name",
            display_name="Collection Name",
            info="The name of the collection within Astra DB where the vectors will be stored.",
            required=True,
        ),
        MultilineInput(
            name="search_input",
            display_name="Search Input",
        ),
        DataInput(
            name="ingest_data",
            display_name="Ingest Data",
            is_list=True,
        ),
        StrInput(
            name="keyspace",
            display_name="Keyspace",
            info="Optional keyspace within Astra DB to use for the collection.",
            advanced=True,
        ),
        DropdownInput(
            name="embedding_choice",
            display_name="Embedding Model or Astra Vectorize",
            info="Determines whether to use Astra Vectorize for the collection.",
            options=["Embedding Model", "Astra Vectorize"],
            real_time_refresh=True,
            value="Embedding Model",
        ),
        HandleInput(
            name="embedding_model",
            display_name="Embedding Model",
            input_types=["Embeddings"],
            info="Allows an embedding model configuration.",
        ),
        DropdownInput(
            name="metric",
            display_name="Metric",
            info="Optional distance metric for vector comparisons in the vector store.",
            options=["cosine", "dot_product", "euclidean"],
            value="cosine",
            advanced=True,
        ),
        IntInput(
            name="batch_size",
            display_name="Batch Size",
            info="Optional number of data to process in a single batch.",
            advanced=True,
        ),
        IntInput(
            name="bulk_insert_batch_concurrency",
            display_name="Bulk Insert Batch Concurrency",
            info="Optional concurrency level for bulk insert operations.",
            advanced=True,
        ),
        IntInput(
            name="bulk_insert_overwrite_concurrency",
            display_name="Bulk Insert Overwrite Concurrency",
            info="Optional concurrency level for bulk insert operations that overwrite existing data.",
            advanced=True,
        ),
        IntInput(
            name="bulk_delete_concurrency",
            display_name="Bulk Delete Concurrency",
            info="Optional concurrency level for bulk delete operations.",
            advanced=True,
        ),
        DropdownInput(
            name="setup_mode",
            display_name="Setup Mode",
            info="Configuration mode for setting up the vector store, with options like 'Sync' or 'Off'.",
            options=["Sync", "Off"],
            advanced=True,
            value="Sync",
        ),
        BoolInput(
            name="pre_delete_collection",
            display_name="Pre Delete Collection",
            info="Boolean flag to determine whether to delete the collection before creating a new one.",
            advanced=True,
        ),
        StrInput(
            name="metadata_indexing_include",
            display_name="Metadata Indexing Include",
            info="Optional list of metadata fields to include in the indexing.",
            is_list=True,
            advanced=True,
        ),
        StrInput(
            name="metadata_indexing_exclude",
            display_name="Metadata Indexing Exclude",
            info="Optional list of metadata fields to exclude from the indexing.",
            is_list=True,
            advanced=True,
        ),
        StrInput(
            name="collection_indexing_policy",
            display_name="Collection Indexing Policy",
            info='Optional JSON string for the "indexing" field of the collection. '
            "See https://docs.datastax.com/en/astra-db-serverless/api-reference/collections.html#the-indexing-option",
            advanced=True,
        ),
        IntInput(
            name="number_of_results",
            display_name="Number of Results",
            info="Number of results to return.",
            advanced=True,
            value=4,
        ),
        DropdownInput(
            name="search_type",
            display_name="Search Type",
            info="Search type to use",
            options=["Similarity", "Similarity with score threshold", "MMR (Max Marginal Relevance)"],
            value="Similarity",
            advanced=True,
        ),
        FloatInput(
            name="search_score_threshold",
            display_name="Search Score Threshold",
            info="Minimum similarity score threshold for search results. "
            "(when using 'Similarity with score threshold')",
            value=0,
            advanced=True,
        ),
        NestedDictInput(
            name="advanced_search_filter",
            display_name="Search Metadata Filter",
            info="Optional dictionary of filters to apply to the search query.",
            advanced=True,
        ),
        DictInput(
            name="search_filter",
            display_name="[DEPRECATED] Search Metadata Filter",
            info="Deprecated: use advanced_search_filter. Optional dictionary of filters to apply to the search query.",
            advanced=True,
            is_list=True,
        ),
    ]

    def del_fields(self, build_config, field_list):
        for field in field_list:
            if field in build_config:
                del build_config[field]

        return build_config

    def insert_in_dict(self, build_config, field_name, new_parameters):
        # Insert the new key-value pair after the found key
        for new_field_name, new_parameter in new_parameters.items():
            # Get all the items as a list of tuples (key, value)
            items = list(build_config.items())

            # Find the index of the key to insert after
            idx = len(items)
            for i, (key, _) in enumerate(items):
                if key == field_name:
                    idx = i + 1
                    break

            items.insert(idx, (new_field_name, new_parameter))

            # Clear the original dictionary and update with the modified items
            build_config.clear()
            build_config.update(items)

        return build_config

    def update_providers_mapping(self):
        # If we don't have token or api_endpoint, we can't fetch the list of providers
        if not self.token or not self.api_endpoint:
            self.log("Astra DB token and API endpoint are required to fetch the list of Vectorize providers.")

            return self.VECTORIZE_PROVIDERS_MAPPING

        try:
            self.log("Dynamically updating list of Vectorize providers.")

            # Get the admin object
            client = DataAPIClient(token=self.token)
            admin = client.get_admin()

            # Get the embedding providers
            db_admin = admin.get_database_admin(self.api_endpoint)
            embedding_providers = db_admin.find_embedding_providers().as_dict()

            vectorize_providers_mapping = {}

            # Map the provider display name to the provider key and models
            for provider_key, provider_data in embedding_providers["embeddingProviders"].items():
                display_name = provider_data["displayName"]
                models = [model["name"] for model in provider_data["models"]]

                vectorize_providers_mapping[display_name] = [provider_key, models]

            # Sort the resulting dictionary
            return defaultdict(list, dict(sorted(vectorize_providers_mapping.items())))
        except Exception as e:  # noqa: BLE001
            self.log(f"Error fetching Vectorize providers: {e}")

            return self.VECTORIZE_PROVIDERS_MAPPING

    def update_build_config(self, build_config: dict, field_value: str, field_name: str | None = None):
        if field_name == "embedding_choice":
            if field_value == "Astra Vectorize":
                self.del_fields(build_config, ["embedding_model"])

                # Update the providers mapping
                vectorize_providers = self.update_providers_mapping()

                new_parameter = DropdownInput(
                    name="embedding_provider",
                    display_name="Embedding Provider",
                    options=vectorize_providers.keys(),
                    value="",
                    required=True,
                    real_time_refresh=True,
                ).to_dict()

                self.insert_in_dict(build_config, "embedding_choice", {"embedding_provider": new_parameter})
            else:
                self.del_fields(
                    build_config,
                    [
                        "embedding_provider",
                        "model",
                        "z_01_model_parameters",
                        "z_02_api_key_name",
                        "z_03_provider_api_key",
                        "z_04_authentication",
                    ],
                )

                new_parameter = HandleInput(
                    name="embedding_model",
                    display_name="Embedding Model",
                    input_types=["Embeddings"],
                    info="Allows an embedding model configuration.",
                ).to_dict()

                self.insert_in_dict(build_config, "embedding_choice", {"embedding_model": new_parameter})

        elif field_name == "embedding_provider":
            self.del_fields(
                build_config,
                ["model", "z_01_model_parameters", "z_02_api_key_name", "z_03_provider_api_key", "z_04_authentication"],
            )

            # Update the providers mapping
            vectorize_providers = self.update_providers_mapping()
            model_options = vectorize_providers[field_value][1]

            new_parameter = DropdownInput(
                name="model",
                display_name="Model",
                info="The embedding model to use for the selected provider. Each provider has a different set of "
                "models available (full list at "
                "https://docs.datastax.com/en/astra-db-serverless/databases/embedding-generation.html):\n\n"
                f"{', '.join(model_options)}",
                options=model_options,
                value=None,
                required=True,
                real_time_refresh=True,
            ).to_dict()

            self.insert_in_dict(build_config, "embedding_provider", {"model": new_parameter})

        elif field_name == "model":
            self.del_fields(
                build_config,
                ["z_01_model_parameters", "z_02_api_key_name", "z_03_provider_api_key", "z_04_authentication"],
            )

            new_parameter_1 = DictInput(
                name="z_01_model_parameters",
                display_name="Model Parameters",
                is_list=True,
            ).to_dict()

            new_parameter_2 = MessageTextInput(
                name="z_02_api_key_name",
                display_name="API Key Name",
                info="The name of the embeddings provider API key stored on Astra. "
                "If set, it will override the 'ProviderKey' in the authentication parameters.",
            ).to_dict()

            new_parameter_3 = SecretStrInput(
                load_from_db=False,
                name="z_03_provider_api_key",
                display_name="Provider API Key",
                info="An alternative to the Astra Authentication that passes an API key for the provider "
                "with each request to Astra DB. "
                "This may be used when Vectorize is configured for the collection, "
                "but no corresponding provider secret is stored within Astra's key management system.",
            ).to_dict()

            new_parameter_4 = DictInput(
                name="z_04_authentication",
                display_name="Authentication Parameters",
                is_list=True,
            ).to_dict()

            self.insert_in_dict(
                build_config,
                "model",
                {
                    "z_01_model_parameters": new_parameter_1,
                    "z_02_api_key_name": new_parameter_2,
                    "z_03_provider_api_key": new_parameter_3,
                    "z_04_authentication": new_parameter_4,
                },
            )

        return build_config

    def build_vectorize_options(self, **kwargs):
        for attribute in [
            "embedding_provider",
            "model",
            "z_01_model_parameters",
            "z_02_api_key_name",
            "z_03_provider_api_key",
            "z_04_authentication",
        ]:
            if not hasattr(self, attribute):
                setattr(self, attribute, None)

        # Fetch values from kwargs if any self.* attributes are None
        provider_value = self.VECTORIZE_PROVIDERS_MAPPING.get(self.embedding_provider, [None])[0] or kwargs.get(
            "embedding_provider"
        )
        model_name = self.model or kwargs.get("model")
        authentication = {**(self.z_04_authentication or kwargs.get("z_04_authentication", {}))}
        parameters = self.z_01_model_parameters or kwargs.get("z_01_model_parameters", {})

        # Set the API key name if provided
        api_key_name = self.z_02_api_key_name or kwargs.get("z_02_api_key_name")
        provider_key = self.z_03_provider_api_key or kwargs.get("z_03_provider_api_key")
        if api_key_name:
            authentication["providerKey"] = api_key_name

        # Set authentication and parameters to None if no values are provided
        if not authentication:
            authentication = None
        if not parameters:
            parameters = None

        return {
            # must match astrapy.info.CollectionVectorServiceOptions
            "collection_vector_service_options": {
                "provider": provider_value,
                "modelName": model_name,
                "authentication": authentication,
                "parameters": parameters,
            },
            "collection_embedding_api_key": provider_key,
        }

    @check_cached_vector_store
    def build_vector_store(self, vectorize_options=None):
        try:
            from langchain_astradb import AstraDBVectorStore
            from langchain_astradb.utils.astradb import SetupMode
        except ImportError as e:
            msg = (
                "Could not import langchain Astra DB integration package. "
                "Please install it with `pip install langchain-astradb`."
            )
            raise ImportError(msg) from e

        try:
            if not self.setup_mode:
                self.setup_mode = self._inputs["setup_mode"].options[0]

            setup_mode_value = SetupMode[self.setup_mode.upper()]
        except KeyError as e:
            msg = f"Invalid setup mode: {self.setup_mode}"
            raise ValueError(msg) from e

        if self.embedding_choice == "Embedding Model":
            embedding_dict = {"embedding": self.embedding_model}
        else:
            from astrapy.info import CollectionVectorServiceOptions

            # Fetch values from kwargs if any self.* attributes are None
            dict_options = vectorize_options or self.build_vectorize_options()

            # Set the embedding dictionary
            embedding_dict = {
                "collection_vector_service_options": CollectionVectorServiceOptions.from_dict(
                    dict_options.get("collection_vector_service_options")
                ),
                "collection_embedding_api_key": dict_options.get("collection_embedding_api_key"),
            }

        try:
            vector_store = AstraDBVectorStore(
                collection_name=self.collection_name,
                token=self.token,
                api_endpoint=self.api_endpoint,
                namespace=self.keyspace or None,
                environment=parse_api_endpoint(self.api_endpoint).environment if self.api_endpoint else None,
                metric=self.metric or None,
                batch_size=self.batch_size or None,
                bulk_insert_batch_concurrency=self.bulk_insert_batch_concurrency or None,
                bulk_insert_overwrite_concurrency=self.bulk_insert_overwrite_concurrency or None,
                bulk_delete_concurrency=self.bulk_delete_concurrency or None,
                setup_mode=setup_mode_value,
                pre_delete_collection=self.pre_delete_collection,
                metadata_indexing_include=[s for s in self.metadata_indexing_include if s] or None,
                metadata_indexing_exclude=[s for s in self.metadata_indexing_exclude if s] or None,
                collection_indexing_policy=orjson.dumps(self.collection_indexing_policy)
                if self.collection_indexing_policy
                else None,
                **embedding_dict,
            )
        except Exception as e:
            msg = f"Error initializing AstraDBVectorStore: {e}"
            raise ValueError(msg) from e

        self._add_documents_to_vector_store(vector_store)

        return vector_store

    def _add_documents_to_vector_store(self, vector_store) -> None:
        documents = []
        for _input in self.ingest_data or []:
            if isinstance(_input, Data):
                documents.append(_input.to_lc_document())
            else:
                msg = "Vector Store Inputs must be Data objects."
                raise TypeError(msg)

        if documents:
            self.log(f"Adding {len(documents)} documents to the Vector Store.")
            try:
                vector_store.add_documents(documents)
            except Exception as e:
                msg = f"Error adding documents to AstraDBVectorStore: {e}"
                raise ValueError(msg) from e
        else:
            self.log("No documents to add to the Vector Store.")

    def _map_search_type(self) -> str:
        if self.search_type == "Similarity with score threshold":
            return "similarity_score_threshold"
        if self.search_type == "MMR (Max Marginal Relevance)":
            return "mmr"
        return "similarity"

    def _build_search_args(self):
        query = self.search_input if isinstance(self.search_input, str) and self.search_input.strip() else None
        search_filter = (
            {k: v for k, v in self.search_filter.items() if k and v and k.strip()} if self.search_filter else None
        )

        if query:
            args = {
                "query": query,
                "search_type": self._map_search_type(),
                "k": self.number_of_results,
                "score_threshold": self.search_score_threshold,
            }
        elif self.advanced_search_filter or search_filter:
            args = {
                "n": self.number_of_results,
            }
        else:
            return {}

        filter_arg = self.advanced_search_filter or {}

        if search_filter:
            self.log(self.log(f"`search_filter` is deprecated. Use `advanced_search_filter`. Cleaned: {search_filter}"))
            filter_arg.update(search_filter)

        if filter_arg:
            args["filter"] = filter_arg

        return args

    def search_documents(self, vector_store=None) -> list[Data]:
        vector_store = vector_store or self.build_vector_store()

        self.log(f"Search input: {self.search_input}")
        self.log(f"Search type: {self.search_type}")
        self.log(f"Number of results: {self.number_of_results}")

        try:
            search_args = self._build_search_args()
        except Exception as e:
            msg = f"Error in AstraDBVectorStore._build_search_args: {e}"
            raise ValueError(msg) from e

        if not search_args:
            self.log("No search input or filters provided. Skipping search.")
            return []

        docs = []
        search_method = "search" if "query" in search_args else "metadata_search"

        try:
            self.log(f"Calling vector_store.{search_method} with args: {search_args}")
            docs = getattr(vector_store, search_method)(**search_args)
        except Exception as e:
            msg = f"Error performing {search_method} in AstraDBVectorStore: {e}"
            raise ValueError(msg) from e

        self.log(f"Retrieved documents: {len(docs)}")

        data = docs_to_data(docs)
        self.log(f"Converted documents to data: {len(data)}")
        self.status = data
        return data

    def get_retriever_kwargs(self):
        search_args = self._build_search_args()
        return {
            "search_type": self._map_search_type(),
            "search_kwargs": search_args,
        }
from typing import Any

from langflow.base.models.model import LCModelComponent
from langflow.field_typing import LanguageModel
from langflow.inputs import DropdownInput, FloatInput, IntInput, SecretStrInput, StrInput
from langflow.inputs.inputs import HandleInput
from langflow.schema.dotdict import dotdict


class NVIDIAModelComponent(LCModelComponent):
    display_name = "NVIDIA"
    description = "Generates text using NVIDIA LLMs."
    icon = "NVIDIA"

    inputs = [
        *LCModelComponent._base_inputs,
        IntInput(
            name="max_tokens",
            display_name="Max Tokens",
            advanced=True,
            info="The maximum number of tokens to generate. Set to 0 for unlimited tokens.",
        ),
        DropdownInput(
            name="model_name",
            display_name="Model Name",
            advanced=False,
            options=["mistralai/mixtral-8x7b-instruct-v0.1"],
            value="mistralai/mixtral-8x7b-instruct-v0.1",
        ),
        StrInput(
            name="base_url",
            display_name="NVIDIA Base URL",
            value="https://integrate.api.nvidia.com/v1",
            refresh_button=True,
            info="The base URL of the NVIDIA API. Defaults to https://integrate.api.nvidia.com/v1.",
        ),
        SecretStrInput(
            name="nvidia_api_key",
            display_name="NVIDIA API Key",
            info="The NVIDIA API Key.",
            advanced=False,
            value="NVIDIA_API_KEY",
        ),
        FloatInput(name="temperature", display_name="Temperature", value=0.1),
        IntInput(
            name="seed",
            display_name="Seed",
            info="The seed controls the reproducibility of the job.",
            advanced=True,
            value=1,
        ),
        HandleInput(
            name="output_parser",
            display_name="Output Parser",
            info="The parser to use to parse the output of the model",
            advanced=True,
            input_types=["OutputParser"],
        ),
    ]

    def update_build_config(self, build_config: dotdict, field_value: Any, field_name: str | None = None):
        if field_name == "base_url" and field_value:
            try:
                build_model = self.build_model()
                ids = [model.id for model in build_model.available_models]
                build_config["model_name"]["options"] = ids
                build_config["model_name"]["value"] = ids[0]
            except Exception as e:
                msg = f"Error getting model names: {e}"
                raise ValueError(msg) from e
        return build_config

    def build_model(self) -> LanguageModel:  # type: ignore[type-var]
        try:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
        except ImportError as e:
            msg = "Please install langchain-nvidia-ai-endpoints to use the NVIDIA model."
            raise ImportError(msg) from e
        nvidia_api_key = self.nvidia_api_key
        temperature = self.temperature
        model_name: str = self.model_name
        max_tokens = self.max_tokens
        seed = self.seed
        return ChatNVIDIA(
            max_tokens=max_tokens or None,
            model=model_name,
            base_url=self.base_url,
            api_key=nvidia_api_key,
            temperature=temperature or 0.1,
            seed=seed,
        )
from langflow.base.io.chat import ChatComponent
from langflow.inputs import BoolInput
from langflow.io import DropdownInput, MessageInput, MessageTextInput, Output
from langflow.schema.message import Message
from langflow.schema.properties import Source
from langflow.utils.constants import MESSAGE_SENDER_AI, MESSAGE_SENDER_NAME_AI, MESSAGE_SENDER_USER


class ChatOutput(ChatComponent):
    display_name = "Chat Output"
    description = "Display a chat message in the Playground."
    icon = "MessagesSquare"
    name = "ChatOutput"

    inputs = [
        MessageInput(
            name="input_value",
            display_name="Text",
            info="Message to be passed as output.",
        ),
        BoolInput(
            name="should_store_message",
            display_name="Store Messages",
            info="Store the message in the history.",
            value=True,
            advanced=True,
        ),
        DropdownInput(
            name="sender",
            display_name="Sender Type",
            options=[MESSAGE_SENDER_AI, MESSAGE_SENDER_USER],
            value=MESSAGE_SENDER_AI,
            advanced=True,
            info="Type of sender.",
        ),
        MessageTextInput(
            name="sender_name",
            display_name="Sender Name",
            info="Name of the sender.",
            value=MESSAGE_SENDER_NAME_AI,
            advanced=True,
        ),
        MessageTextInput(
            name="session_id",
            display_name="Session ID",
            info="The session ID of the chat. If empty, the current session ID parameter will be used.",
            advanced=True,
        ),
        MessageTextInput(
            name="data_template",
            display_name="Data Template",
            value="{text}",
            advanced=True,
            info="Template to convert Data to Text. If left empty, it will be dynamically set to the Data's text key.",
        ),
        MessageTextInput(
            name="background_color",
            display_name="Background Color",
            info="The background color of the icon.",
            advanced=True,
        ),
        MessageTextInput(
            name="chat_icon",
            display_name="Icon",
            info="The icon of the message.",
            advanced=True,
        ),
        MessageTextInput(
            name="text_color",
            display_name="Text Color",
            info="The text color of the name",
            advanced=True,
        ),
    ]
    outputs = [
        Output(
            display_name="Message",
            name="message",
            method="message_response",
        ),
    ]

    def _build_source(self, _id: str | None, display_name: str | None, source: str | None) -> Source:
        source_dict = {}
        if _id:
            source_dict["id"] = _id
        if display_name:
            source_dict["display_name"] = display_name
        if source:
            source_dict["source"] = source
        return Source(**source_dict)

    def message_response(self) -> Message:
        _source, _icon, _display_name, _source_id = self.get_properties_from_source_component()
        _background_color = self.background_color
        _text_color = self.text_color
        if self.chat_icon:
            _icon = self.chat_icon
        message = self.input_value if isinstance(self.input_value, Message) else Message(text=self.input_value)
        message.sender = self.sender
        message.sender_name = self.sender_name
        message.session_id = self.session_id
        message.flow_id = self.graph.flow_id if hasattr(self, "graph") else None
        message.properties.source = self._build_source(_source_id, _display_name, _source)
        message.properties.icon = _icon
        message.properties.background_color = _background_color
        message.properties.text_color = _text_color
        if self.session_id and isinstance(message, Message) and self.should_store_message:
            stored_message = self.send_message(
                message,
            )
            self.message.value = stored_message
            message = stored_message

        self.status = message
        return message
