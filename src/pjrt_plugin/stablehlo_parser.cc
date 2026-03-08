// MLIR-based StableHLO parser
// Parses StableHLO bytecode/text and keeps MLIR alive for direct execution

#include "pjrt_plugin/stablehlo_parser.h"

#include <unordered_set>

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/Serialization.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/VhloOps.h"

namespace mps {

namespace {

// Register all dialects needed for StableHLO parsing
void registerDialects(mlir::MLIRContext& context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::stablehlo::StablehloDialect>();
    registry.insert<mlir::vhlo::VhloDialect>();
    registry.insert<mlir::chlo::ChloDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
    // Allow unknown dialects (e.g., sdy/Shardy for sharding) to pass through
    context.allowUnregisteredDialects();
}

// Run the inliner pass to inline all func.call operations
bool runInlinerPass(mlir::MLIRContext& context, mlir::ModuleOp module) {
    // Mark all non-main functions as private so they can be inlined
    module.walk([&](mlir::func::FuncOp funcOp) {
        if (funcOp.getName() != "main" && funcOp.isPublic()) {
            funcOp.setPrivate();
        }
    });

    mlir::PassManager pm(&context);
    pm.addPass(mlir::createInlinerPass());

    // Ignore errors from inliner - we'll handle func.call at runtime
    (void)pm.run(module);
    return true;
}

// Find the entry function (usually "main")
mlir::func::FuncOp findEntryFunction(mlir::ModuleOp module) {
    mlir::func::FuncOp entry = nullptr;
    module.walk([&](mlir::func::FuncOp funcOp) {
        if (funcOp.getName() == "main" || !entry) {
            entry = funcOp;
        }
    });
    return entry;
}

// Common parsing logic
ParsedModule finalizeModule(std::unique_ptr<mlir::MLIRContext> context,
                            mlir::OwningOpRef<mlir::ModuleOp> module) {
    ParsedModule result;

    if (!module) {
        return result;
    }

    if (!runInlinerPass(*context, *module)) {
        return result;
    }

    mlir::func::FuncOp entry = findEntryFunction(*module);
    if (!entry) {
        return result;
    }

    // Op support is checked at execution time, not parse time
    result.unsupported_ops = {};

    result.context = std::move(context);
    result.module = std::move(module);
    result.entry_func = entry;

    return result;
}

}  // namespace

ParsedModule parseStableHLOBytecode(const char* data, size_t size) {
    auto context = std::make_unique<mlir::MLIRContext>();
    registerDialects(*context);

    auto buffer = llvm::MemoryBuffer::getMemBuffer(llvm::StringRef(data, size),
                                                   "stablehlo_bytecode",
                                                   /*RequiresNullTerminator=*/false);

    mlir::OwningOpRef<mlir::ModuleOp> moduleOp =
        mlir::stablehlo::deserializePortableArtifact(buffer->getBuffer(), context.get());

    if (!moduleOp) {
        llvm::SourceMgr sourceMgr;
        sourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());
        moduleOp = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, context.get());
    }

    return finalizeModule(std::move(context), std::move(moduleOp));
}

ParsedModule parseStableHLOText(const std::string& text) {
    auto context = std::make_unique<mlir::MLIRContext>();
    registerDialects(*context);

    mlir::OwningOpRef<mlir::ModuleOp> moduleOp =
        mlir::parseSourceString<mlir::ModuleOp>(text, context.get());

    return finalizeModule(std::move(context), std::move(moduleOp));
}

std::string bytecodeToText(const char* data, size_t size) {
    mlir::MLIRContext context;
    registerDialects(context);

    auto buffer =
        llvm::MemoryBuffer::getMemBuffer(llvm::StringRef(data, size), "stablehlo_bytecode", false);

    mlir::OwningOpRef<mlir::ModuleOp> moduleOp =
        mlir::stablehlo::deserializePortableArtifact(buffer->getBuffer(), &context);

    if (!moduleOp) {
        llvm::SourceMgr sourceMgr;
        auto bufferCopy = llvm::MemoryBuffer::getMemBuffer(llvm::StringRef(data, size),
                                                           "stablehlo_bytecode", false);
        sourceMgr.AddNewSourceBuffer(std::move(bufferCopy), llvm::SMLoc());
        moduleOp = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
    }

    if (!moduleOp) {
        return "";
    }

    std::string result;
    llvm::raw_string_ostream os(result);
    moduleOp->print(os);
    return result;
}

}  // namespace mps
