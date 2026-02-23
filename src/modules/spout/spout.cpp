#include "spout.h"
#include <core/module_dependencies.h>
#include "consumer/spout_consumer.h"
#include "producer/spout_producer.h"

namespace caspar { namespace spout {

void init(const core::module_dependencies& dependencies)
{
    dependencies.producer_registry->register_producer_factory(L"SPOUT", create_spout_producer);
    dependencies.consumer_registry->register_consumer_factory(L"SPOUT", create_spout_consumer);
}

}}
