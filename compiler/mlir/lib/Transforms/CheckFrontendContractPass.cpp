#include "pyc/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace pyc {
namespace {

class CheckFrontendContractPass : public PassWrapper<CheckFrontendContractPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CheckFrontendContractPass)

  StringRef getArgument() const override { return "pyc-check-frontend-contract"; }
  StringRef getDescription() const override {
    return "Verify required frontend contract attrs are present and match the supported contract";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool ok = true;

    auto emitModule = [&](llvm::StringRef code, llvm::StringRef msg, llvm::StringRef hint) {
      auto d = module.emitError();
      d << "[" << code << "] " << msg;
      if (!hint.empty())
        d << " (hint: " << hint << ")";
    };

    static constexpr const char *kRequiredContract = "pycircuit";
    auto modContract = module->getAttrOfType<StringAttr>("pyc.frontend.contract");
    if (!modContract) {
      emitModule("PYC901", "missing required module attr `pyc.frontend.contract`",
                 "regenerate .pyc with the current pyCircuit frontend and keep module attrs intact");
      ok = false;
    } else if (modContract.getValue() != kRequiredContract) {
      auto d = module.emitError();
      d << "[PYC902] frontend contract mismatch: expected `" << kRequiredContract << "`, got `"
        << modContract.getValue() << "` (hint: regenerate .pyc with matching toolchain)";
      ok = false;
    }

    module.walk([&](func::FuncOp f) {
      auto checkStrAttr = [&](StringRef name, llvm::StringRef code, llvm::StringRef hint) -> StringAttr {
        auto attr = f->getAttrOfType<StringAttr>(name);
        if (!attr) {
          auto d = f.emitError();
          d << "[" << code << "] missing required func attr `" << name << "`";
          if (!hint.empty())
            d << " (hint: " << hint << ")";
          ok = false;
        }
        return attr;
      };

      auto checkArrAttr = [&](StringRef name, llvm::StringRef code, llvm::StringRef hint) -> ArrayAttr {
        auto attr = f->getAttrOfType<ArrayAttr>(name);
        if (!attr) {
          auto d = f.emitError();
          d << "[" << code << "] missing required func attr `" << name << "`";
          if (!hint.empty())
            d << " (hint: " << hint << ")";
          ok = false;
        }
        return attr;
      };

      auto kind = checkStrAttr("pyc.kind", "PYC903", "frontend must stamp symbol kind metadata");
      auto inl = checkStrAttr("pyc.inline", "PYC904", "frontend must stamp inline metadata");
      (void)checkStrAttr("pyc.params", "PYC905", "frontend must stamp canonical specialization params");
      (void)checkStrAttr("pyc.base", "PYC906", "frontend must stamp canonical base symbol name");
      (void)checkArrAttr("arg_names", "PYC907", "frontend must stamp canonical port names");
      (void)checkArrAttr("result_names", "PYC908", "frontend must stamp canonical port names");

      if (kind) {
        auto k = kind.getValue();
        if (k != "module" && k != "function" && k != "template") {
          f.emitError() << "[PYC909] invalid `pyc.kind` value: " << k
                        << " (hint: allowed values are module/function/template)";
          ok = false;
        }
      }

      if (inl) {
        auto v = inl.getValue();
        if (v != "true" && v != "false") {
          f.emitError() << "[PYC910] invalid `pyc.inline` value: " << v
                        << " (hint: allowed values are true|false)";
          ok = false;
        }
      }
    });

    if (!ok)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createCheckFrontendContractPass() {
  return std::make_unique<CheckFrontendContractPass>();
}

static PassRegistration<CheckFrontendContractPass> pass;

} // namespace pyc

