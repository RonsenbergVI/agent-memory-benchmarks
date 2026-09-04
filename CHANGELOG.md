# Changelog

## [0.1.0](https://github.com/RonsenbergVI/agent-memory-benchmarks/compare/v0.0.1...v0.1.0) (2026-09-04)


### Features

* **agentmemory:** add agentmemory integration ([#118](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/118)) ([b3e16bf](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/b3e16bf8292827ee2119ef4e47b65a59c414438c))
* **cognee:** add cognee integration ([#113](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/113)) ([b258d06](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/b258d067bc9b2572302bfffdd58492af0166f458))
* distinguish partial token accounting from full and from none ([#124](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/124)) ([3129e34](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/3129e34e112144af26811f878846b2daa1c91fba))
* **everos:** add everos integration ([#121](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/121)) ([61832f8](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/61832f8431e0c448e8f98b1411a01ff952947b19))
* **hindsight:** add hindsight integration ([#120](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/120)) ([78337e8](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/78337e8761c2904a310fe171c404118bc47868ef))
* **letta:** add agent-driven ingestion that exercises the LLM ([#122](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/122)) ([6ad2d55](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/6ad2d559142ad3f4ce092a082fc0e386f0eb7a6c))
* mark systems whose token spend cannot be observed ([#123](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/123)) ([4c157e1](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/4c157e198a66884d8d5e3b570879a3e386254ad6))


### Bug fixes

* count tokens spent off the worker thread, and async chat tokens ([#119](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/119)) ([2a324c8](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/2a324c87728a3cdae6f938c59cc04633aba8967e))
* **fraise:** phrase-only recall, SDK 0.1.0b3, and no session header ([#102](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/102)) ([4833c21](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/4833c21d2bd8a3e4697df21aa03081c12a79634f))
* **graphiti:** document the raised llm token ceiling ([#90](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/90)) ([3302d5f](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/3302d5fb560f8499c799946540365711688ed409))
* **graphiti:** drop truststore from the TLS path ([#100](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/100)) ([c177b95](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/c177b95452611a1e3cec53391fb3615a1dae5181))
* **graphiti:** stop closing async clients in teardown ([#94](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/94)) ([a8d5fa4](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/a8d5fa4604d5df20748039189ec1e25c62500dd7))
* **letta:** stop injecting a session header into stored passages ([#104](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/104)) ([299374b](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/299374b3c36a24c6aeec8304e12f09731d23a110))
* match tagged toml values in release-please lockfile filters ([#92](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/92)) ([02bb9ef](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/02bb9ef245130100d8897b75925b68705ca4b80d))
* **mem0:** serialize client setup to avoid the qdrant collection race ([#103](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/103)) ([437a48f](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/437a48fa5b61b803fa130b2813503188fb38a98a))


### Dependencies

* bump actions/download-artifact from 4 to 8 ([#5](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/5)) ([fcc4acb](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/fcc4acb27218902ed85718d61de5166f0b4a1140))
* bump actions/upload-artifact from 4 to 7 ([#6](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/6)) ([a88e482](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/a88e4827d25a4cecf631ed4430b0d6e2e55eca39))
* bump docker/login-action from 3 to 4 ([#8](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/8)) ([72458c3](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/72458c3a7b5dcdf2bd6eaeffaa6edf8e80d9b636))
* bump everalgo-boundary from 0.2.1 to 0.3.0 ([#144](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/144)) ([8bebd3d](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/8bebd3d4d9f7626eb93483a2fc7704171f73a097))
* bump huggingface-hub from 1.27.0 to 1.28.0 ([#110](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/110)) ([229af82](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/229af8245364cc0f7212c785270dbc28dcd5ef39))
* bump openai from 2.54.0 to 3.1.0 ([#7](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/7)) ([38e9d7e](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/38e9d7e87edbd28c89430b0d7d9af409da68eb2a))
* bump openai from 3.1.0 to 3.2.0 ([#85](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/85)) ([de3ef85](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/de3ef85dcab056fa6ef192b3a6cd019759dc8942))
* bump pydantic from 2.13.4 to 2.13.5 ([#146](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/146)) ([715928f](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/715928ff98d8a2bf43fa075a1d801bf8edde3996))
* bump pydantic-ai from 2.31.0 to 2.31.1 ([#37](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/37)) ([980ce8a](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/980ce8a4cc898b8fe624cf3537586bc957244e05))
* bump pydantic-ai from 2.31.1 to 2.33.0 ([#112](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/112)) ([802ad6a](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/802ad6ad064f38905a87f0e63e6b3c58f1d9bfdc))
* bump stefanzweifel/git-auto-commit-action from 5 to 7 ([e843a8d](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/e843a8d66443acd82db677db7503c3ac5406f94a))
* bump stefanzweifel/git-auto-commit-action from 5 to 7 ([#11](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/11)) ([d189daa](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/d189daa2535e5e514b4de8f0be973941542716be))
* bump tiktoken from 0.13.0 to 0.14.0 ([#38](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/38)) ([1aaced6](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/1aaced64d7147c79a71aa28ed331c865e9ed99b0))


### Maintenance

* add citation ([#101](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/101)) ([e2634a7](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/e2634a71580f93faee0cf89789b1d97bc08aba9f))
* add dataset tiering ([#115](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/115)) ([1cf79ec](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/1cf79ecf0f95f702a3892771a9e2a1c9d02a5fd4))
* add missing benchmark status badges ([#136](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/136)) ([b6b4d7e](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/b6b4d7ef59d786c44b5c30c4b1ccf71b3c51ac42))
* add missing openai api key ([#137](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/137)) ([3254a0e](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/3254a0e90e34b49279cbb5af5994aa55e3d88b15))
* add run freshness in plots ([#88](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/88)) ([322c1a9](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/322c1a9e0efd0ce586e82a21a6e923b2168ee321))
* add sponsorship documents ([#114](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/114)) ([1224626](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/1224626f5db2c4c79624f5b6ae4e9fbddc0188d7))
* CI run improvements ([#96](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/96)) ([d9accd3](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/d9accd30e779ebaf84afe4d22ec411af1d32fa86))
* drop component from release tags ([#17](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/17)) ([2f50cea](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/2f50cea27773e4d52e6dd7a3e63ddd63fb622f53))
* **fraise:** move to v0.1.0-rc.1 and SDK 0.1.0b4 ([#127](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/127)) ([77ad453](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/77ad453e6209dc06d7b669a4c9705ca639ed7601))
* **fraise:** update fraise docker image version ([#106](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/106)) ([6b9b41d](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/6b9b41d7efb027f8f19c5ef6633661d300c6d84b))
* **main:** release agentmemory 0.1.0 ([#133](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/133)) ([036610f](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/036610fb1ca58bc04aeb67c44d13fae216194d9a))
* **main:** release agentmemory 0.2.0 ([#139](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/139)) ([53c5be4](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/53c5be452869c151464b27ec55b40ac4d8963f4c))
* **main:** release cognee 0.1.0 ([#132](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/132)) ([fb9cab0](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/fb9cab0c36eb06a29b7d551a0e4f8df4be464d6c))
* **main:** release everos 0.1.0 ([#135](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/135)) ([9e60001](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/9e6000106388ef5754e2aba75887254656deadf5))
* **main:** release fraise 0.1.0 ([#36](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/36)) ([7b845e7](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/7b845e7b3074e72e61571e0701f7292311ca88d2))
* **main:** release fraise 0.2.0 ([#128](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/128)) ([1996c55](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/1996c556da562303f4d7528f945589dc9f170dbb))
* **main:** release fraise 0.3.0 ([#143](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/143)) ([67a0972](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/67a0972fc939cb8e65e626ecb811b97e7650865f))
* **main:** release graphiti 0.1.0 ([#34](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/34)) ([bc22102](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/bc2210225335dc6ae877f303246250ddc3d4bc22))
* **main:** release graphiti 0.2.0 ([#93](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/93)) ([d4ce825](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/d4ce825406c98252a5ad0cdda40ab1985ba765d4))
* **main:** release graphiti 0.3.0 ([#95](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/95)) ([cbb3f28](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/cbb3f281b9dc52e2bf332c563a97c0a09e9eced4))
* **main:** release hindsight 0.1.0 ([#134](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/134)) ([32b12cc](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/32b12ccf3903019099bedc988593f222b1c6016a))
* **main:** release hindsight 0.2.0 ([#138](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/138)) ([990a991](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/990a99106ce5e061df2ad48a0cd35c4a04136646))
* **main:** release letta 0.1.0 ([#33](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/33)) ([bfbb05d](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/bfbb05d9a9df7afd41ade05d0123f89d5872b206))
* **main:** release letta 0.2.0 ([#105](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/105)) ([28aace6](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/28aace65210e2c672db6d54fbbd77f39e8057072))
* **main:** release mem0 0.1.0 ([#35](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/35)) ([f068ae4](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/f068ae4ac8417d9392254dbf9d338ddf3e5a5aaa))
* **main:** release mem0 0.2.0 ([#109](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/109)) ([fe5da80](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/fe5da80ee1bb690a4f941ed9fee4584f6b49ac32))
* pause LongMemEval across the benchmarked systems ([#126](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/126)) ([bcfd888](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/bcfd8882ed903f84f945c23200265f1c68bbd108))
* polish readme ([#86](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/86)) ([65a0715](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/65a0715ed1ac990be69886ab539b7b15e7514880))
* preparing first full run ([#19](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/19)) ([cd91b4b](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/cd91b4b9767fdcbac970a92b03e50963d3ddfc45))
* remove individual memory dependabot ([#39](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/39)) ([a28f25b](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/a28f25bb603e43bf9bb9f1c8f6438927c852f80d))
* repo hygiene - first full run prep ([#31](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/31)) ([2a1bc04](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/2a1bc0484b69d8ee3fcec69dbf6e4c793e857a75))
* update run artefact saving ([#98](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/98)) ([87064e5](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/87064e55ab9106ce71b2aa24046287fa837dc5db))
* upgrade fraise version and use one fraise graph per conversation ([#142](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues/142)) ([8865abb](https://github.com/RonsenbergVI/agent-memory-benchmarks/commit/8865abb2d8289f7b67b652604df98bb97b3af877))
